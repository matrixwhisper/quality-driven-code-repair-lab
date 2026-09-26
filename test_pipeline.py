"""CPU-only regression tests; no model download or GPU is required."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent import AgentSettings, repair_task
from config import ExperimentConfig
from dataset import (
	TASKS,
	assert_validation_only,
	dataset_fingerprint,
	split_tasks,
	validate_behavior,
	validate_task_records,
)
from evaluation import aggregate_results
from models import GenerationResult, strip_code_fence
from sandbox import run_candidate
from search import hill_climb_validation

TRAIN_TASKS = split_tasks(seed=2026)["train"]


class DatasetTests(unittest.TestCase):
	def test_reviewed_tasks_are_structurally_complete(self) -> None:
		self.assertEqual(validate_task_records(), [])
		self.assertEqual(len(TASKS), 10)
		self.assertTrue(all(task.review_status == "accepted" for task in TASKS))

	def test_references_pass_and_known_seeds_fail(self) -> None:
		pretest_tasks = split_tasks(seed=2026)["train"] + split_tasks(seed=2026)["validation"]
		self.assertEqual(validate_behavior(pretest_tasks), [])

	def test_split_is_deterministic_and_family_disjoint(self) -> None:
		first = split_tasks(seed=7)
		second = split_tasks(seed=7)
		self.assertEqual(first, second)
		families = {name: {task.family for task in partition} for name, partition in first.items()}
		self.assertFalse(families["train"] & families["validation"])
		self.assertFalse(families["train"] & families["test"])
		self.assertFalse(families["validation"] & families["test"])
		self.assertTrue(all(first[name] for name in first))

	def test_dataset_fingerprint_changes_with_content(self) -> None:
		original = dataset_fingerprint()
		modified = TASKS[:-1]
		self.assertNotEqual(original, dataset_fingerprint(modified))

	def test_quality_checker_flags_vague_and_contradictory_specs(self) -> None:
		vague = replace(TRAIN_TASKS[0], specification="Clamp values as appropriate, etc.")
		vague_problems = validate_task_records((vague,))
		self.assertTrue(any("vague language" in problem for problem in vague_problems))
		contradictory = replace(
			TRAIN_TASKS[0],
			specification="Bounds are inclusive and exclusive at the same time.",
		)
		contradiction_problems = validate_task_records((contradictory,))
		self.assertTrue(any("both inclusive and exclusive" in problem for problem in contradiction_problems))

	def test_quality_checker_flags_coverage_without_a_named_test(self) -> None:
		task = replace(TRAIN_TASKS[0], coverage_tags=("test_case_that_does_not_exist", "inside_and_edges"))
		problems = validate_task_records((task,))
		self.assertTrue(any("no matching named test case" in problem for problem in problems))

	def test_test_split_cannot_be_used_for_tuning(self) -> None:
		with self.assertRaises(PermissionError):
			assert_validation_only("test")
		assert_validation_only("validation")


class SandboxTests(unittest.TestCase):
	def test_reference_passes(self) -> None:
		task = TRAIN_TASKS[0]
		result = run_candidate(task.reference_code, task.tests_code, timeout_seconds=3)
		self.assertTrue(result.compile_success)
		self.assertTrue(result.tests_passed, result.output)

	def test_seed_is_caught_by_task_tests(self) -> None:
		task = TRAIN_TASKS[0]
		result = run_candidate(task.starter_code, task.tests_code, timeout_seconds=3)
		self.assertTrue(result.compile_success)
		self.assertFalse(result.tests_passed)
		self.assertEqual(result.failure_kind, "assertion_failure")

	def test_compile_errors_and_no_test_suites_are_distinguished(self) -> None:
		task = TASKS[0]
		invalid = run_candidate("def broken(:\n", task.tests_code)
		self.assertFalse(invalid.compile_success)
		self.assertEqual(invalid.failure_kind, "compile_error")
		no_tests = run_candidate(task.reference_code, "value = 1\n")
		self.assertTrue(no_tests.compile_success)
		self.assertFalse(no_tests.tests_passed)
		self.assertEqual(no_tests.failure_kind, "no_tests_discovered")

	def test_candidate_timeout_is_reported(self) -> None:
		task = TRAIN_TASKS[0]
		function_header = task.reference_code.splitlines()[0]
		infinite_candidate = f"{function_header}\n    while True:\n        pass\n"
		result = run_candidate(infinite_candidate, task.tests_code, timeout_seconds=0.25)
		self.assertEqual(result.failure_kind, "timeout")
		self.assertFalse(result.tests_passed)

	def test_output_is_capped(self) -> None:
		task = TRAIN_TASKS[0]
		code = task.starter_code
		tests = "import unittest\nfrom solution import clamp\nclass RepairTests(unittest.TestCase):\n    def test_output(self):\n        print('x' * 50000)\n        self.assertEqual(clamp(2, 0, 3), 4)\n"
		result = run_candidate(code, tests, timeout_seconds=3)
		self.assertLessEqual(len(result.output.encode("utf-8")), 20_000)


class FakeGenerator:
	def __init__(self, outputs: list[str]) -> None:
		self.outputs = outputs
		self.prompts: list[str] = []

	def generate(self, prompt: str, **kwargs) -> GenerationResult:
		self.prompts.append(prompt)
		text = self.outputs.pop(0)
		return GenerationResult(text, 0.01, 20, 10, 1000, 1200, None)


class AgentTests(unittest.TestCase):
	def test_retry_includes_test_feedback_and_recovers(self) -> None:
		task = TRAIN_TASKS[0]
		model = FakeGenerator([task.starter_code, task.reference_code])
		result = repair_task(task, model, AgentSettings(max_retries=1, sandbox_timeout_seconds=3))
		self.assertTrue(result.passed)
		self.assertEqual(result.attempt_count, 2)
		self.assertEqual(result.attempts[0].failure_kind, "assertion_failure")
		self.assertIn("AssertionError", result.attempts[0].diagnostic_output)
		self.assertIn("Previous test feedback", model.prompts[1])

	def test_retry_limit_is_enforced(self) -> None:
		task = TRAIN_TASKS[0]
		model = FakeGenerator([task.starter_code, task.starter_code])
		result = repair_task(task, model, AgentSettings(max_retries=1, sandbox_timeout_seconds=3))
		self.assertFalse(result.passed)
		self.assertEqual(result.attempt_count, 2)

	def test_markdown_fences_are_removed(self) -> None:
		self.assertEqual(strip_code_fence("```python\nx = 1\n```"), "x = 1\n")


class SearchAndMetricsTests(unittest.TestCase):
	def test_search_rejects_non_validation_split(self) -> None:
		with self.assertRaises(PermissionError):
			hill_climb_validation((), FakeGenerator([]), AgentSettings(), "test")

	def test_aggregate_reports_required_metrics(self) -> None:
		task = TASKS[0]
		model = FakeGenerator([task.reference_code])
		outcome = repair_task(task, model, AgentSettings(max_retries=0, sandbox_timeout_seconds=3))
		result = aggregate_results([outcome], "fake", "unit_test")
		self.assertEqual(result.pass_rate, 1.0)
		self.assertEqual(result.compile_success_rate, 1.0)
		self.assertEqual(result.pass_after_retries_rate, 1.0)
		self.assertGreaterEqual(result.mean_generation_latency_seconds, 0.0)


class ConfigTests(unittest.TestCase):
	def test_config_round_trip(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "nested" / "config.json"
			original = ExperimentConfig(run_dir=Path("runs"), seed=123)
			original.save(path)
			restored = ExperimentConfig.load(path)
			self.assertEqual(restored.seed, 123)
			self.assertEqual(restored.model_id, original.model_id)

	def test_invalid_split_fractions_are_rejected(self) -> None:
		with self.assertRaises(ValueError):
			ExperimentConfig(train_fraction=0.8, validation_fraction=0.3)


if __name__ == "__main__":
	unittest.main(verbosity=2)
