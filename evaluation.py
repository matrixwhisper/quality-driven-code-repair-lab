"""Per-task evaluation, aggregate metrics, and readable experiment reports."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agent import AgentSettings, CodeGenerator, RepairOutcome, repair_task
from dataset import RepairTask


@dataclass(frozen=True)
class EvaluationResult:
	model_label: str
	split_name: str
	task_count: int
	pass_rate: float
	compile_success_rate: float
	first_attempt_test_success_rate: float
	pass_after_retries_rate: float
	mean_attempts: float
	mean_generation_latency_seconds: float
	p95_generation_latency_seconds: float
	mean_model_process_rss_bytes: float | None
	max_model_process_rss_bytes: int | None
	mean_sandbox_peak_rss_bytes: float | None
	max_sandbox_peak_rss_bytes: int | None
	max_cuda_allocated_bytes: int | None
	failure_counts: dict[str, int]
	failure_examples: dict[str, list[str]]
	per_task: tuple[dict[str, object], ...]

	def to_dict(self) -> dict[str, object]:
		return asdict(self)


def evaluation_from_dict(data: dict[str, object]) -> EvaluationResult:
	"""Rehydrate a persisted aggregate result without rerunning inference."""
	values = dict(data)
	values["per_task"] = tuple(values["per_task"])
	return EvaluationResult(**values)  # type: ignore[arg-type]


def _percentile95(values: list[float]) -> float:
	if not values:
		return 0.0
	ordered = sorted(values)
	return ordered[max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999) - 1))]


def aggregate_results(
	outcomes: Iterable[RepairOutcome],
	model_label: str,
	split_name: str,
) -> EvaluationResult:
	items = tuple(outcomes)
	if not items:
		raise ValueError("cannot aggregate an empty evaluation")
	passed = sum(outcome.passed for outcome in items)
	first_compile = sum(bool(outcome.attempts and outcome.attempts[0].compile_success) for outcome in items)
	first_pass = sum(bool(outcome.attempts and outcome.attempts[0].tests_passed) for outcome in items)
	generation_latencies = [
		attempt.generation_latency_seconds
		for outcome in items
		for attempt in outcome.attempts
		if attempt.generation_latency_seconds >= 0
	]
	process_rss = [
		value
		for outcome in items
		for attempt in outcome.attempts
		for value in (attempt.process_rss_before_bytes, attempt.process_rss_after_bytes)
		if value is not None
	]
	sandbox_rss = [
		attempt.sandbox_peak_rss_bytes
		for outcome in items
		for attempt in outcome.attempts
		if attempt.sandbox_peak_rss_bytes is not None
	]
	gpu_rss = [
		attempt.cuda_peak_allocated_bytes
		for outcome in items
		for attempt in outcome.attempts
		if attempt.cuda_peak_allocated_bytes is not None
	]
	failure_counts: Counter[str] = Counter()
	failure_examples: dict[str, list[str]] = {}
	for outcome in items:
		if not outcome.passed:
			failure = outcome.attempts[-1].failure_kind or "unknown"
			failure_counts[failure] += 1
			failure_examples.setdefault(failure, []).append(outcome.task_id)
	return EvaluationResult(
		model_label=model_label,
		split_name=split_name,
		task_count=len(items),
		pass_rate=passed / len(items),
		compile_success_rate=first_compile / len(items),
		first_attempt_test_success_rate=first_pass / len(items),
		pass_after_retries_rate=passed / len(items),
		mean_attempts=statistics.fmean(outcome.attempt_count for outcome in items),
		mean_generation_latency_seconds=statistics.fmean(generation_latencies) if generation_latencies else 0.0,
		p95_generation_latency_seconds=_percentile95(generation_latencies),
		mean_model_process_rss_bytes=statistics.fmean(process_rss) if process_rss else None,
		max_model_process_rss_bytes=max(process_rss) if process_rss else None,
		mean_sandbox_peak_rss_bytes=statistics.fmean(sandbox_rss) if sandbox_rss else None,
		max_sandbox_peak_rss_bytes=max(sandbox_rss) if sandbox_rss else None,
		max_cuda_allocated_bytes=max(gpu_rss) if gpu_rss else None,
		failure_counts=dict(sorted(failure_counts.items())),
		failure_examples={key: value[:10] for key, value in sorted(failure_examples.items())},
		per_task=tuple(outcome.to_dict() for outcome in items),
	)


def evaluate_tasks(
	tasks: Iterable[RepairTask],
	model: CodeGenerator,
	model_label: str,
	split_name: str,
	settings: AgentSettings | None = None,
) -> EvaluationResult:
	outcomes = [repair_task(task, model, settings) for task in tasks]
	return aggregate_results(outcomes, model_label, split_name)


def save_evaluation(result: EvaluationResult, run_dir: str | Path) -> tuple[Path, Path]:
	destination = Path(run_dir)
	destination.mkdir(parents=True, exist_ok=True)
	safe_label = "".join(character if character.isalnum() or character in "-_" else "_" for character in result.model_label)
	prefix = f"{safe_label}_{result.split_name}"
	json_path = destination / f"{prefix}.json"
	json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
	report_path = destination / f"{prefix}_report.md"
	report_path.write_text(format_report(result), encoding="utf-8")
	return json_path, report_path


def format_report(result: EvaluationResult) -> str:
	rss_mb = "not measured" if result.mean_model_process_rss_bytes is None else f"{result.mean_model_process_rss_bytes / 1024**2:.1f} MiB mean sampled model-process RSS"
	sandbox_rss_mb = "not measured" if result.mean_sandbox_peak_rss_bytes is None else f"{result.mean_sandbox_peak_rss_bytes / 1024**2:.1f} MiB mean sampled sandbox peak RSS"
	gpu_mb = "not measured" if result.max_cuda_allocated_bytes is None else f"{result.max_cuda_allocated_bytes / 1024**2:.1f} MiB peak CUDA allocation"
	lines = [
		f"# Evaluation: {result.model_label}",
		"",
		f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
		f"- Split: `{result.split_name}`; tasks: {result.task_count}",
		"- Status: measured by this run; do not interpret absent runs as zero performance.",
		f"- Pass rate after retries: {result.pass_after_retries_rate:.1%}",
		f"- First-attempt compile success: {result.compile_success_rate:.1%}",
		f"- First-attempt test success: {result.first_attempt_test_success_rate:.1%}",
		f"- Mean attempts per task: {result.mean_attempts:.2f}",
		f"- Mean generation latency per attempt: {result.mean_generation_latency_seconds:.3f}s",
		f"- P95 generation latency per attempt: {result.p95_generation_latency_seconds:.3f}s",
		f"- Resource use: {rss_mb}; {sandbox_rss_mb}; {gpu_mb}.",
		"",
		"## Final failure analysis",
	]
	if result.failure_counts:
		for kind, count in result.failure_counts.items():
			examples = ", ".join(result.failure_examples.get(kind, []))
			lines.append(f"- `{kind}`: {count} task(s); examples: {examples}")
	else:
		lines.append("- No final failures in this split.")
	lines.extend([
		"",
		"## Interpretation limits",
		"This report describes only the tasks in the named split and this exact run. "
		"It does not establish general code-repair ability. Inspect the dataset review "
		"notes and run manifest; report hardware and model revision when comparing runs.",
		"",
	])
	return "\n".join(lines)
