"""Validation-only hill climbing for repair-agent settings."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from agent import AgentSettings, CodeGenerator
from dataset import RepairTask, assert_validation_only
from evaluation import EvaluationResult, evaluate_tasks


def _score(result: EvaluationResult) -> tuple[float, float, float, float]:
	"""Favor held-out task success, then first-pass success, then lower cost."""
	return (
		result.pass_after_retries_rate,
		result.first_attempt_test_success_rate,
		-result.mean_generation_latency_seconds,
		-result.mean_attempts,
	)


def _neighbors(settings: AgentSettings) -> list[AgentSettings]:
	candidates = [
		replace(settings, max_retries=max(0, settings.max_retries - 1)),
		replace(settings, max_retries=min(3, settings.max_retries + 1)),
		replace(settings, feedback_chars=max(0, settings.feedback_chars - 400)),
		replace(settings, feedback_chars=min(3200, settings.feedback_chars + 400)),
		replace(settings, max_new_tokens=max(128, settings.max_new_tokens - 128)),
		replace(settings, max_new_tokens=min(768, settings.max_new_tokens + 128)),
		replace(settings, temperature=0.0 if settings.temperature > 0 else 0.2),
	]
	unique: list[AgentSettings] = []
	seen: set[tuple[object, ...]] = set()
	for candidate in candidates:
		key = tuple(asdict(candidate).values())
		if candidate != settings and key not in seen:
			unique.append(candidate)
			seen.add(key)
	return unique


def hill_climb_validation(
	tasks: tuple[RepairTask, ...],
	model: CodeGenerator,
	initial_settings: AgentSettings,
	split_name: str,
	evaluation_budget: int = 5,
	output_path: str | Path | None = None,
) -> tuple[AgentSettings, list[dict[str, object]]]:
	"""Search neighbors on validation data only; test inputs are never accepted."""
	assert_validation_only(split_name)
	if not tasks:
		raise ValueError("validation task set must not be empty")
	if evaluation_budget < 1:
		raise ValueError("evaluation_budget must be at least one")
	if any(task.review_status != "accepted" for task in tasks):
		raise ValueError("search accepts only reviewed tasks")

	current = initial_settings
	history: list[dict[str, object]] = []
	current_result: EvaluationResult | None = None
	tested: set[tuple[object, ...]] = set()
	for step in range(evaluation_budget):
		proposals = [current] if step == 0 else _neighbors(current)
		proposal = next(
			(item for item in proposals if tuple(asdict(item).values()) not in tested),
			None,
		)
		if proposal is None:
			break
		tested.add(tuple(asdict(proposal).values()))
		result = evaluate_tasks(tasks, model, f"search_step_{step:02d}", "validation", proposal)
		history.append({
			"step": step,
			"split": "validation",
			"task_ids": [task.task_id for task in tasks],
			"settings": asdict(proposal),
			"metrics": result.to_dict(),
			"score": list(_score(result)),
		})
		if current_result is None or _score(result) > _score(current_result):
			current = proposal
			current_result = result

	if current_result is None:
		raise RuntimeError("search performed no evaluations")
	if output_path is not None:
		destination = Path(output_path)
		destination.parent.mkdir(parents=True, exist_ok=True)
		destination.write_text(json.dumps({
			"search_split": "validation",
			"validation_task_ids": [task.task_id for task in tasks],
			"test_accessed": False,
			"evaluation_budget": evaluation_budget,
			"selected_settings": asdict(current),
			"selected_validation_metrics": current_result.to_dict(),
			"history": history,
		}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return current, history
