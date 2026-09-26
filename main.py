"""Command-line entry point for the staged Quality-Driven Code Repair Lab."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent import AgentSettings
from config import ExperimentConfig, seed_everything
from dataset import (
	REVIEW_LEDGER,
	TASKS,
	dataset_fingerprint,
	split_manifest,
	split_tasks,
	validate_behavior,
	validate_task_records,
)
from evaluation import EvaluationResult, evaluate_tasks, evaluation_from_dict, save_evaluation
from finetuning import train_qlora
from models import HFCodeModel
from search import hill_climb_validation


def _settings(config: ExperimentConfig) -> AgentSettings:
	return AgentSettings(
		max_retries=config.max_retries,
		feedback_chars=config.feedback_chars,
		max_new_tokens=config.max_new_tokens,
		temperature=config.temperature,
		top_p=config.top_p,
		sandbox_timeout_seconds=config.sandbox_timeout_seconds,
	)


def _model(config: ExperimentConfig, adapter_path: str | None = None) -> HFCodeModel:
	return HFCodeModel(
		model_id=config.model_id,
		revision=config.revision,
		adapter_path=adapter_path,
	)


def _save_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_config(run_dir: Path) -> ExperimentConfig:
	saved_path = run_dir / "config.json"
	if saved_path.is_file():
		config = ExperimentConfig.load(saved_path)
		config.run_dir = run_dir
		return config
	return ExperimentConfig(run_dir=run_dir)


def _runtime_manifest(config: ExperimentConfig, model_metadata: dict[str, object] | None = None) -> dict[str, object]:
	versions: dict[str, str] = {}
	for package in ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "psutil"):
		try:
			versions[package] = importlib.metadata.version(package)
		except importlib.metadata.PackageNotFoundError:
			versions[package] = "not installed"
	return {
		"python_version": sys.version,
		"platform": platform.platform(),
		"cpu_count": os.cpu_count(),
		"package_versions": versions,
		"github_commit": os.environ.get("GITHUB_SHA"),
		"config": config.to_dict(),
		"dataset_sha256": dataset_fingerprint(),
		"model": model_metadata,
	}


def run_quality(config: ExperimentConfig) -> dict[str, object]:
	structural = validate_task_records()
	if structural:
		raise ValueError("Structural dataset checks failed:\n- " + "\n- ".join(structural))
	splits = split_tasks(TASKS, config.seed, config.train_fraction, config.validation_fraction)
	pretest_tasks = splits["train"] + splits["validation"]
	behavioral = validate_behavior(pretest_tasks)
	if behavioral:
		raise ValueError("Reference/seed behavioral checks failed:\n- " + "\n- ".join(behavioral))
	manifest = split_manifest(seed=config.seed, train_fraction=config.train_fraction, validation_fraction=config.validation_fraction)
	manifest["task_count"] = len(TASKS)
	manifest["pretest_behavior_check_task_ids"] = [task.task_id for task in pretest_tasks]
	manifest["test_behavior_checked"] = False
	manifest["review_ledger"] = list(REVIEW_LEDGER)
	manifest["dataset_limitations"] = [
		"Small, original, hand-authored pilot corpus; results are not statistically representative.",
		"Family-level holdout reduces near-duplicate leakage but cannot establish cross-project generalization.",
		"The test split is locally readable; access is procedurally gated by the CLI, not cryptographically hidden.",
		"Unit tests are finite behavioral checks and may miss plausible specification violations.",
	]
	_save_json(config.run_dir / "dataset_manifest.json", manifest)
	_save_json(config.run_dir / "review_ledger.json", {"decisions": list(REVIEW_LEDGER)})
	_save_json(config.run_dir / "runtime_manifest.json", _runtime_manifest(config))
	return manifest


def run_baseline_validation(config: ExperimentConfig) -> EvaluationResult:
	seed_everything(config.seed)
	config.save(config.run_dir / "config.json")
	model = _model(config)
	metadata = model.metadata()
	if config.revision in {"main", "master"}:
		config.revision = str(metadata["resolved_revision"])
		config.save(config.run_dir / "config.json")
	_save_json(config.run_dir / "runtime_manifest.json", _runtime_manifest(config, metadata))
	tasks = split_tasks(TASKS, config.seed, config.train_fraction, config.validation_fraction)["validation"]
	result = evaluate_tasks(tasks, model, "base", "validation", _settings(config))
	save_evaluation(result, config.run_dir)
	return result


def run_training(config: ExperimentConfig) -> Path:
	splits = split_tasks(TASKS, config.seed, config.train_fraction, config.validation_fraction)
	training_quality_problems = validate_behavior(splits["train"])
	if training_quality_problems:
		raise ValueError("Training split quality checks failed:\n- " + "\n- ".join(training_quality_problems))
	adapter_path = train_qlora(config, splits["train"], config.run_dir / "adapter")
	manifest = json.loads((adapter_path / "adapter_manifest.json").read_text(encoding="utf-8"))
	config.revision = str(manifest["base_model_revision"])
	config.save(config.run_dir / "config.json")
	return adapter_path


def run_search(
	config: ExperimentConfig,
	adapter_path: str | None,
	output_path: Path | None = None,
) -> tuple[AgentSettings, list[dict[str, object]]]:
	seed_everything(config.seed)
	path = output_path or config.run_dir / "search.json"
	validation_tasks = split_tasks(
		TASKS, config.seed, config.train_fraction, config.validation_fraction,
	)["validation"]
	model = _model(config, adapter_path)
	metadata = model.metadata()
	best, history = hill_climb_validation(
		validation_tasks,
		model,
		_settings(config),
		split_name="validation",
		evaluation_budget=config.search_evaluation_budget,
		output_path=path,
	)
	search_record = json.loads(path.read_text(encoding="utf-8"))
	search_record["dataset_sha256"] = dataset_fingerprint()
	search_record["model_metadata"] = metadata
	search_record["adapter_path"] = str(Path(adapter_path).resolve()) if adapter_path else None
	_save_json(path, search_record)
	selected_metrics = dict(search_record["selected_validation_metrics"])
	selected_metrics["model_label"] = "qlora_adapter" if adapter_path else "base"
	validation_result = evaluation_from_dict(selected_metrics)
	save_evaluation(validation_result, config.run_dir)
	return best, history


def _load_approved_search(config: ExperimentConfig, adapter_path: str, search_path: Path) -> tuple[AgentSettings, dict[str, Any]]:
	if not search_path.is_file():
		raise FileNotFoundError("Final test evaluation requires a completed validation search artifact")
	record = json.loads(search_path.read_text(encoding="utf-8"))
	expected = split_tasks(TASKS, config.seed, config.train_fraction, config.validation_fraction)
	expected_ids = [task.task_id for task in expected["validation"]]
	expected_train_ids = [task.task_id for task in expected["train"]]
	if record.get("search_split") != "validation" or record.get("test_accessed") is not False:
		raise PermissionError("Search artifact is not marked as validation-only")
	if record.get("dataset_sha256") != dataset_fingerprint():
		raise ValueError("Dataset changed after search; rerun validation search before final evaluation")
	if record.get("validation_task_ids") != expected_ids:
		raise ValueError("Search artifact validation task ids do not match this split")
	if any(entry.get("split") != "validation" for entry in record.get("history", [])):
		raise PermissionError("Search history contains a non-validation evaluation")
	if record.get("adapter_path") != str(Path(adapter_path).resolve()):
		raise ValueError("Final adapter path differs from the adapter tuned on validation")
	manifest_path = Path(adapter_path) / "adapter_manifest.json"
	if not manifest_path.is_file():
		raise FileNotFoundError(f"Saved adapter manifest is missing: {manifest_path}")
	adapter_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	if adapter_manifest.get("dataset_sha256") != dataset_fingerprint():
		raise ValueError("Adapter was trained against a different dataset revision")
	if adapter_manifest.get("split_used") != "train" or adapter_manifest.get("test_accessed") is not False:
		raise PermissionError("Adapter manifest does not prove train-only fine-tuning")
	if adapter_manifest.get("train_task_ids") != expected_train_ids:
		raise ValueError("Adapter training task ids do not match this train split")
	if adapter_manifest.get("base_model_id") != config.model_id:
		raise ValueError("Adapter base model does not match the experiment config")
	if adapter_manifest.get("base_model_revision") != config.revision:
		raise ValueError("Adapter base revision does not match the pinned experiment config")
	metadata = record.get("model_metadata") or {}
	if metadata.get("model_id") != config.model_id or metadata.get("resolved_revision") != config.revision:
		raise ValueError("Validation search used a different base model revision")
	return AgentSettings(**record["selected_settings"]), adapter_manifest


def run_final_comparison(config: ExperimentConfig, adapter_path: str, search_path: Path) -> dict[str, object]:
	final_result_path = config.run_dir / "final_comparison.json"
	test_lock_path = config.run_dir / "final_test_started.json"
	if final_result_path.exists() or test_lock_path.exists():
		raise FileExistsError(
			f"Final test evaluation already started in {config.run_dir}; refusing an accidental repeat"
		)
	settings, adapter_manifest = _load_approved_search(config, adapter_path, search_path)
	seed_everything(config.seed)
	test_tasks = split_tasks(
		TASKS, config.seed, config.train_fraction, config.validation_fraction,
	)["test"]
	if set(task.task_id for task in test_tasks) & set(
		json.loads(search_path.read_text(encoding="utf-8"))["validation_task_ids"]
	):
		raise AssertionError("test and validation task ids overlap")
	_save_json(test_lock_path, {
		"status": "started",
		"dataset_sha256": dataset_fingerprint(),
		"test_task_ids": [task.task_id for task in test_tasks],
		"search_artifact": str(search_path.resolve()),
		"adapter_path": str(Path(adapter_path).resolve()),
		"note": "Presence of this file blocks repeat test access, including after an interrupted run.",
	})
	test_quality_problems = validate_behavior(test_tasks)
	if test_quality_problems:
		raise ValueError("Final test fixture checks failed:\n- " + "\n- ".join(test_quality_problems))
	seed_everything(config.seed)
	base = _model(config)
	base_metadata = base.metadata()
	base_result = evaluate_tasks(test_tasks, base, "base", "test", settings)
	save_evaluation(base_result, config.run_dir)
	del base
	seed_everything(config.seed)
	tuned = _model(config, adapter_path)
	tuned_metadata = tuned.metadata()
	tuned_result = evaluate_tasks(test_tasks, tuned, "qlora_adapter", "test", settings)
	save_evaluation(tuned_result, config.run_dir)
	comparison = {
		"status": "measured_final_test",
		"test_accessed": True,
		"dataset_sha256": dataset_fingerprint(),
		"test_task_ids": [task.task_id for task in test_tasks],
		"search_artifact": str(search_path.resolve()),
		"adapter_manifest": str((Path(adapter_path) / "adapter_manifest.json").resolve()),
		"adapter_train_task_ids": adapter_manifest["train_task_ids"],
		"base_model_metadata": base_metadata,
		"adapter_model_metadata": tuned_metadata,
		"selected_agent_settings": asdict(settings),
		"base": base_result.to_dict(),
		"qlora_adapter": tuned_result.to_dict(),
		"pass_rate_delta_adapter_minus_base": tuned_result.pass_rate - base_result.pass_rate,
		"limitations": [
			"Small pilot test set; each task changes the pass rate substantially.",
			"This is one seed/model revision and does not establish statistical significance.",
			"Local procedural holdout is not equivalent to a hidden external benchmark.",
		],
	}
	_save_json(final_result_path, comparison)
	report = [
		"# Final held-out comparison",
		"",
		"Status: measured by this run. Both models were evaluated after validation-only tuning, with the same tasks, prompts, decoding settings, and retry policy.",
		"",
		"| Metric | Base model | QLoRA adapter |",
		"|---|---:|---:|",
		f"| Pass rate after retries | {base_result.pass_rate:.1%} | {tuned_result.pass_rate:.1%} |",
		f"| First-attempt compile success | {base_result.compile_success_rate:.1%} | {tuned_result.compile_success_rate:.1%} |",
		f"| First-attempt test success | {base_result.first_attempt_test_success_rate:.1%} | {tuned_result.first_attempt_test_success_rate:.1%} |",
		f"| Mean attempts | {base_result.mean_attempts:.2f} | {tuned_result.mean_attempts:.2f} |",
		f"| Mean generation latency / attempt | {base_result.mean_generation_latency_seconds:.3f}s | {tuned_result.mean_generation_latency_seconds:.3f}s |",
		"",
		f"Pass-rate delta (adapter minus base): {comparison['pass_rate_delta_adapter_minus_base']:+.1%}.",
		"",
		"These results apply only to the small, hand-authored task set. Review per-task JSON and failure reports before drawing conclusions. No unrun experiment is represented as measured.",
		"",
	]
	(config.run_dir / "final_comparison.md").write_text("\n".join(report), encoding="utf-8")
	return comparison


def run_all(config: ExperimentConfig) -> dict[str, object]:
	print("[1/5] Checking dataset quality")
	run_quality(config)
	print("[2/5] Evaluating base model on validation only")
	baseline = run_baseline_validation(config)
	print(json.dumps({"base_validation_pass_rate": baseline.pass_rate}))
	print("[3/5] Training QLoRA adapter on training split")
	adapter_path = run_training(config)
	try:
		import torch
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	except ImportError:
		pass
	print("[4/5] Hill-climbing adapter settings on validation only")
	run_search(config, str(adapter_path))
	print("[5/5] Opening held-out test for final base/adapter comparison")
	return run_final_comparison(config, str(adapter_path), config.run_dir / "search.json")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Quality-Driven Code Repair Lab")
	parser.add_argument("command", choices=("quality", "baseline", "train", "search", "final", "all"))
	parser.add_argument("--run-dir", type=Path, default=Path("runs"))
	parser.add_argument("--adapter-path", type=str, default=None)
	parser.add_argument("--search-results", type=Path, default=None)
	return parser.parse_args()


def main() -> int:
	args = _parse_args()
	config = _load_config(args.run_dir)
	try:
		if args.command == "quality":
			print(json.dumps(run_quality(config), indent=2, sort_keys=True))
		elif args.command == "baseline":
			result = run_baseline_validation(config)
			print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
		elif args.command == "train":
			print(run_training(config))
		elif args.command == "search":
			if not args.adapter_path:
				raise ValueError("search requires --adapter-path")
			run_search(config, args.adapter_path)
			print(args.run_dir / "search.json")
		elif args.command == "final":
			if not args.adapter_path:
				raise ValueError("final requires --adapter-path")
			search_path = args.search_results or args.run_dir / "search.json"
			print(json.dumps(run_final_comparison(config, args.adapter_path, search_path), indent=2, sort_keys=True))
		elif args.command == "all":
			print(json.dumps(run_all(config), indent=2, sort_keys=True))
	except (ValueError, RuntimeError, PermissionError, FileNotFoundError, FileExistsError) as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
