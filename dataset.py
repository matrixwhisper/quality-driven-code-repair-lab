"""Reviewed, self-contained Python repair tasks and deterministic splits.

The initial benchmark is deliberately small and original: tasks are written in
this repository so their provenance and licenses are unambiguous. A small suite
is useful for validating the pipeline, but not large enough for broad claims.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Literal


Difficulty = Literal["easy", "medium"]
ReviewStatus = Literal["accepted", "revised", "rejected"]
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class RepairTask:
	task_id: str
	family: str
	title: str
	specification: str
	starter_code: str
	reference_code: str
	tests_code: str
	difficulty: Difficulty
	review_status: ReviewStatus
	review_notes: str
	provenance: str = "Original, hand-authored benchmark task; no third-party task source."
	coverage_tags: tuple[str, ...] = ()

	def prompt(self, feedback: str | None = None) -> str:
		"""Render the stable model prompt; feedback is reserved for repair retries."""
		retry = f"\nPrevious test feedback:\n{feedback}\n" if feedback else ""
		return (
			"Repair the Python module below to satisfy the specification. "
			"Return only the complete replacement Python module, with no markdown.\n\n"
			f"Task: {self.title}\nSpecification:\n{self.specification}\n\n"
			f"Current module:\n{self.starter_code}\n"
			f"{retry}\nReplacement module:"
		)


def _task(
	task_id: str,
	family: str,
	title: str,
	specification: str,
	starter: str,
	reference: str,
	tests: str,
	difficulty: Difficulty,
	tags: tuple[str, ...],
	note: str = "Specification, reference, and tests reviewed together; edge cases are explicit.",
) -> RepairTask:
	return RepairTask(
		task_id=task_id,
		family=family,
		title=title,
		specification=specification.strip(),
		starter_code=starter.strip() + "\n",
		reference_code=reference.strip() + "\n",
		tests_code=tests.strip() + "\n",
		difficulty=difficulty,
		review_status="accepted",
		review_notes=note,
		coverage_tags=tags,
	)


TASKS: tuple[RepairTask, ...] = (
	_task(
		"clamp_bounds", "numeric_bounds", "Clamp a number to inclusive bounds",
		"Implement clamp(value, lower, upper). Return value bounded inclusively by lower and upper. Raise ValueError when lower > upper.",
		"def clamp(value, lower, upper):\n    return max(value, lower)\n",
		"def clamp(value, lower, upper):\n    if lower > upper:\n        raise ValueError('lower must not exceed upper')\n    return min(max(value, lower), upper)\n",
		"import unittest\nfrom solution import clamp\n\nclass RepairTests(unittest.TestCase):\n    def test_inside_and_edges(self):\n        self.assertEqual(clamp(4, 0, 9), 4)\n        self.assertEqual(clamp(0, 0, 9), 0)\n    def test_both_outside(self):\n        self.assertEqual(clamp(-3, 0, 9), 0)\n        self.assertEqual(clamp(12, 0, 9), 9)\n    def test_invalid_bounds(self):\n        with self.assertRaises(ValueError): clamp(1, 5, 2)\n\nif __name__ == '__main__': unittest.main()",
		"easy", ("inside_and_edges", "both_outside", "invalid_bounds"),
	),
	_task(
		"safe_divide", "numeric_arithmetic", "Raise on division by zero",
		"Implement safe_divide(numerator, denominator). Return ordinary Python division. A zero denominator must raise ZeroDivisionError, not return a sentinel.",
		"def safe_divide(numerator, denominator):\n    return 0 if denominator == 0 else numerator / denominator\n",
		"def safe_divide(numerator, denominator):\n    if denominator == 0:\n        raise ZeroDivisionError('division by zero')\n    return numerator / denominator\n",
		"import unittest\nfrom solution import safe_divide\n\nclass RepairTests(unittest.TestCase):\n    def test_integer_and_fractional_results(self):\n        self.assertEqual(safe_divide(8, 2), 4)\n        self.assertEqual(safe_divide(1, 4), 0.25)\n    def test_zero_denominator(self):\n        with self.assertRaises(ZeroDivisionError): safe_divide(2, 0)\n\nif __name__ == '__main__': unittest.main()",
		"easy", ("integer_and_fractional_results", "zero_denominator"),
	),
	_task(
		"normalize_whitespace", "text_normalization", "Normalize whitespace",
		"Implement normalize_whitespace(text). Collapse each run of Unicode whitespace to one ASCII space and remove leading/trailing whitespace. Empty or all-whitespace input returns an empty string.",
		"def normalize_whitespace(text):\n    return text.strip()\n",
		"def normalize_whitespace(text):\n    return ' '.join(text.split())\n",
		"import unittest\nfrom solution import normalize_whitespace\n\nclass RepairTests(unittest.TestCase):\n    def test_internal_runs_and_unicode_whitespace(self):\n        self.assertEqual(normalize_whitespace('  alpha\\t\\nbeta  '), 'alpha beta')\n        self.assertEqual(normalize_whitespace('a\\u00a0b'), 'a b')\n    def test_empty_values(self):\n        self.assertEqual(normalize_whitespace(''), '')\n        self.assertEqual(normalize_whitespace(' \\t\\n'), '')\n\nif __name__ == '__main__': unittest.main()",
		"easy", ("internal_runs_and_unicode_whitespace", "empty_values"),
	),
	_task(
		"stable_dedupe", "sequence_ordering", "Deduplicate without changing order",
		"Implement dedupe_preserve_order(items). Keep the first occurrence of every hashable item, preserve input order, and return a list.",
		"def dedupe_preserve_order(items):\n    return list(set(items))\n",
		"def dedupe_preserve_order(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n",
		"import unittest\nfrom solution import dedupe_preserve_order\n\nclass RepairTests(unittest.TestCase):\n    def test_order_and_first_occurrence(self):\n        self.assertEqual(dedupe_preserve_order([3, 1, 3, 2, 1]), [3, 1, 2])\n    def test_empty_input_and_return_type(self):\n        result = dedupe_preserve_order([])\n        self.assertEqual(result, [])\n        self.assertIsInstance(result, list)\n\nif __name__ == '__main__': unittest.main()",
		"easy", ("order_and_first_occurrence", "empty_input_and_return_type"),
	),
	_task(
		"median_even", "statistics", "Compute the median",
		"Implement median(values). Accept a non-empty sequence of numbers, do not mutate it, return the middle value for odd length and the arithmetic mean of the two middle values for even length. Raise ValueError for empty input.",
		"def median(values):\n    ordered = sorted(values)\n    return ordered[len(ordered) // 2]\n",
		"def median(values):\n    if not values:\n        raise ValueError('median is undefined for an empty sequence')\n    ordered = sorted(values)\n    middle = len(ordered) // 2\n    if len(ordered) % 2:\n        return ordered[middle]\n    return (ordered[middle - 1] + ordered[middle]) / 2\n",
		"import unittest\nfrom solution import median\n\nclass RepairTests(unittest.TestCase):\n    def test_odd_and_even_lengths(self):\n        self.assertEqual(median([9, 1, 5]), 5)\n        self.assertEqual(median([1, 2, 8, 9]), 5.0)\n    def test_unsorted_input_is_not_mutated(self):\n        values = [4, 1, 3, 2]\n        self.assertEqual(median(values), 2.5)\n        self.assertEqual(values, [4, 1, 3, 2])\n    def test_empty_input(self):\n        with self.assertRaises(ValueError): median([])\n\nif __name__ == '__main__': unittest.main()",
		"medium", ("odd_and_even_lengths", "unsorted_input_is_not_mutated", "empty_input"),
	),
	_task(
		"merge_adjacent_intervals", "interval_algorithms", "Merge overlapping or adjacent intervals",
		"Implement merge_intervals(intervals). Each interval is a pair [start, end] with start <= end. Sort by start and merge intervals that overlap or touch (next_start <= current_end). Return a list of two-item lists; empty input returns [].",
		"def merge_intervals(intervals):\n    ordered = sorted(intervals)\n    merged = []\n    for start, end in ordered:\n        if not merged or start > merged[-1][1]:\n            merged.append([start, end])\n        else:\n            merged[-1][1] = max(merged[-1][1], end)\n    return merged\n",
		"def merge_intervals(intervals):\n    ordered = sorted(intervals)\n    merged = []\n    for start, end in ordered:\n        if start > end:\n            raise ValueError('interval start must not exceed end')\n        if not merged or start > merged[-1][1]:\n            merged.append([start, end])\n        else:\n            merged[-1][1] = max(merged[-1][1], end)\n    return merged\n",
		"import unittest\nfrom solution import merge_intervals\n\nclass RepairTests(unittest.TestCase):\n    def test_overlap_adjacency_and_sorting(self):\n        self.assertEqual(merge_intervals([[5, 8], [1, 3], [3, 4], [10, 12]]), [[1, 4], [5, 8], [10, 12]])\n    def test_nested_and_empty(self):\n        self.assertEqual(merge_intervals([[1, 10], [2, 3]]), [[1, 10]])\n        self.assertEqual(merge_intervals([]), [])\n    def test_invalid_interval(self):\n        with self.assertRaises(ValueError): merge_intervals([[4, 2]])\n\nif __name__ == '__main__': unittest.main()",
		"medium", ("overlap_adjacency_and_sorting", "nested_and_empty", "invalid_interval"),
	),
	_task(
		"strict_parse_bool", "input_parsing", "Parse explicit boolean values",
		"Implement parse_bool(value). Return bool values unchanged; accept case-insensitive, trimmed strings 'true'/'yes'/'1' and 'false'/'no'/'0'. Reject every other type or string with ValueError.",
		"def parse_bool(value):\n    return bool(value)\n",
		"def parse_bool(value):\n    if isinstance(value, bool):\n        return value\n    if isinstance(value, str):\n        normalized = value.strip().lower()\n        if normalized in {'true', 'yes', '1'}:\n            return True\n        if normalized in {'false', 'no', '0'}:\n            return False\n    raise ValueError('expected an explicit boolean value')\n",
		"import unittest\nfrom solution import parse_bool\n\nclass RepairTests(unittest.TestCase):\n    def test_boolean_and_accepted_strings(self):\n        for value in (True, 'TRUE', ' yes ', '1'):\n            self.assertIs(parse_bool(value), True)\n        for value in (False, 'False', ' no ', '0'):\n            self.assertIs(parse_bool(value), False)\n    def test_rejects_ambiguous_values(self):\n        for value in ('', 'on', 'maybe', 0, 1, None):\n            with self.subTest(value=value), self.assertRaises(ValueError): parse_bool(value)\n\nif __name__ == '__main__': unittest.main()",
		"easy", ("boolean_and_accepted_strings", "rejects_ambiguous_values"),
	),
	_task(
		"rectangular_transpose", "matrix_operations", "Transpose a rectangular matrix",
		"Implement transpose(matrix). Return a list of rows. Empty input returns []. Reject ragged matrices with ValueError; do not silently truncate rows.",
		"def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\n",
		"def transpose(matrix):\n    if not matrix:\n        return []\n    width = len(matrix[0])\n    if any(len(row) != width for row in matrix):\n        raise ValueError('matrix must be rectangular')\n    return [list(row) for row in zip(*matrix)]\n",
		"import unittest\nfrom solution import transpose\n\nclass RepairTests(unittest.TestCase):\n    def test_rectangular_and_single_row(self):\n        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])\n        self.assertEqual(transpose([[7, 8]]), [[7], [8]])\n    def test_empty_and_ragged(self):\n        self.assertEqual(transpose([]), [])\n        with self.assertRaises(ValueError): transpose([[1, 2], [3]])\n\nif __name__ == '__main__': unittest.main()",
		"medium", ("rectangular_and_single_row", "empty_and_ragged"),
	),
	_task(
		"first_binary_search", "search_algorithms", "Return the first matching binary-search index",
		"Implement binary_search(values, target) for a nondecreasing sequence. Return the lowest index containing target, or -1 when absent. The input is not mutated.",
		"def binary_search(values, target):\n    low, high = 0, len(values) - 1\n    while low <= high:\n        middle = (low + high) // 2\n        if values[middle] == target:\n            return middle\n        if values[middle] < target:\n            low = middle + 1\n        else:\n            high = middle - 1\n    return -1\n",
		"def binary_search(values, target):\n    low, high = 0, len(values)\n    while low < high:\n        middle = (low + high) // 2\n        if values[middle] < target:\n            low = middle + 1\n        else:\n            high = middle\n    return low if low < len(values) and values[low] == target else -1\n",
		"import unittest\nfrom solution import binary_search\n\nclass RepairTests(unittest.TestCase):\n    def test_first_duplicate_and_endpoints(self):\n        self.assertEqual(binary_search([1, 2, 2, 2, 9], 2), 1)\n        self.assertEqual(binary_search([1, 2, 9], 1), 0)\n        self.assertEqual(binary_search([1, 2, 9], 9), 2)\n    def test_absent_and_empty(self):\n        self.assertEqual(binary_search([1, 3], 2), -1)\n        self.assertEqual(binary_search([], 2), -1)\n\nif __name__ == '__main__': unittest.main()",
		"medium", ("first_duplicate_and_endpoints", "absent_and_empty"),
	),
	_task(
		"slugify_text", "text_normalization", "Create a predictable ASCII slug",
		"Implement slugify(text). Lowercase ASCII letters, replace each run of non-ASCII-alphanumeric characters with one hyphen, and strip edge hyphens. Do not transliterate non-ASCII letters.",
		"def slugify(text):\n    return text.lower().replace(' ', '-')\n",
		"import re\n\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n",
		"import unittest\nfrom solution import slugify\n\nclass RepairTests(unittest.TestCase):\n    def test_punctuation_runs_and_edges(self):\n        self.assertEqual(slugify('  Hello,  World!  '), 'hello-world')\n        self.assertEqual(slugify('a___b'), 'a-b')\n    def test_ascii_policy_and_empty(self):\n        self.assertEqual(slugify('cafe'), 'cafe')\n        self.assertEqual(slugify(''), '')\n        self.assertEqual(slugify('éclair'), 'clair')\n\nif __name__ == '__main__': unittest.main()",
		"easy", ("punctuation_runs_and_edges", "ascii_policy_and_empty"),
	),
	_task(
		"chunk_sequence", "sequence_chunking", "Split a sequence into fixed-size chunks",
		"Implement chunked(values, size). Return a list of consecutive list chunks of at most size; preserve order; include a shorter final chunk. Raise ValueError unless size is a positive integer (bool is not accepted). Empty input returns [].",
		"def chunked(values, size):\n    return [list(values[i:i + size]) for i in range(0, len(values), size)]\n",
		"def chunked(values, size):\n    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:\n        raise ValueError('size must be a positive integer')\n    return [list(values[i:i + size]) for i in range(0, len(values), size)]\n",
		"import unittest\nfrom solution import chunked\n\nclass RepairTests(unittest.TestCase):\n    def test_full_and_short_final_chunk(self):\n        self.assertEqual(chunked([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])\n    def test_empty_and_invalid_sizes(self):\n        self.assertEqual(chunked([], 3), [])\n        for size in (0, -1, 1.5, True):\n            with self.subTest(size=size), self.assertRaises(ValueError): chunked([1], size)\n\nif __name__ == '__main__': unittest.main()",
		"easy", ("full_and_short_final_chunk", "empty_and_invalid_sizes"),
	),
)


# Kept outside TASKS: these examples explain review decisions but are not scored.
REVIEW_LEDGER: tuple[dict[str, str], ...] = (
	{"task_id": "stable_dedupe_v0", "decision": "revised", "reason": "The first draft did not say whether first-seen or sorted order was required; specification and tests now require first-seen order."},
	{"task_id": "coerce_anything_to_bool", "decision": "rejected", "reason": "The original request did not define accepted inputs or invalid-value behavior, so multiple incompatible repairs would pass an underspecified task."},
)


class DatasetQualityError(ValueError):
	"""Raised when task records are structurally or behaviorally unsound."""


def validate_task_records(tasks: tuple[RepairTask, ...] = TASKS) -> list[str]:
	"""Run cheap structural checks before any costly model or sandbox work."""
	problems: list[str] = []
	seen_ids: set[str] = set()
	vague_specification = re.compile(
		r"\b(?:etc\.?|and so on|as appropriate|as needed|reasonable|sensible|normally)\b",
		re.IGNORECASE,
	)
	contradiction_pairs = (
		(r"\binclusive(?:ly)?\b", r"\bexclusive(?:ly)?\b", "both inclusive and exclusive boundary rules"),
		(r"\bempty input (?:must )?return\b", r"\bempty input (?:must )?raise\b", "both return and raise requirements for empty input"),
		(r"\bzero denominator (?:must )?return\b", r"\bzero denominator (?:must )?raise\b", "both return and raise requirements for zero denominator"),
		(r"\bmust not mutate\b", r"\bmust mutate\b", "contradictory input-mutation requirements"),
	)
	for task in tasks:
		prefix = f"{task.task_id}: "
		if not re.fullmatch(r"[a-z][a-z0-9_]+", task.task_id):
			problems.append(prefix + "task id is not a stable lowercase identifier")
		if task.task_id in seen_ids:
			problems.append(prefix + "duplicate task id")
		seen_ids.add(task.task_id)
		for field_name in ("family", "title", "specification", "review_notes", "provenance"):
			if not getattr(task, field_name).strip():
				problems.append(prefix + f"missing {field_name}")
		if vague_specification.search(task.specification):
			problems.append(prefix + "specification contains vague language requiring review")
		for left, right, description in contradiction_pairs:
			if re.search(left, task.specification, re.IGNORECASE) and re.search(right, task.specification, re.IGNORECASE):
				problems.append(prefix + f"specification contains {description}")
		if task.review_status != "accepted":
			problems.append(prefix + "scored task is not accepted")
		if len(task.coverage_tags) < 2 or len(set(task.coverage_tags)) != len(task.coverage_tags):
			problems.append(prefix + "coverage tags are missing or duplicated")
		for coverage_tag in task.coverage_tags:
			if coverage_tag not in task.tests_code:
				problems.append(prefix + f"coverage tag {coverage_tag!r} has no matching named test case")
		if not re.search(r"\bdef\s+[A-Za-z_]\w*\s*\(", task.starter_code):
			problems.append(prefix + "starter has no function definition")
		if not re.search(r"\bdef\s+[A-Za-z_]\w*\s*\(", task.reference_code):
			problems.append(prefix + "reference has no function definition")
		if "class RepairTests" not in task.tests_code or "unittest.main" not in task.tests_code:
			problems.append(prefix + "test suite is not a runnable unittest module")
		if len(re.findall(r"\b(?:assertEqual|assertIs|assertTrue|assertFalse|assertRaises)\s*\(", task.tests_code)) < 2:
			problems.append(prefix + "test suite has fewer than two explicit assertions")
		for label, source in (("starter", task.starter_code), ("reference", task.reference_code)):
			try:
				compile(source, f"{task.task_id}_{label}.py", "exec")
			except SyntaxError as exc:
				problems.append(prefix + f"{label} does not compile: {exc.msg}")
		if not task.tests_code.strip():
			problems.append(prefix + "missing behavioral coverage")
	return problems


def dataset_fingerprint(tasks: tuple[RepairTask, ...] = TASKS) -> str:
	"""Hash task content and review metadata for run provenance."""
	payload = [asdict(task) for task in sorted(tasks, key=lambda item: item.task_id)]
	encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def split_tasks(
	tasks: tuple[RepairTask, ...] = TASKS,
	seed: int = 2026,
	train_fraction: float = 0.5,
	validation_fraction: float = 0.25,
) -> dict[SplitName, tuple[RepairTask, ...]]:
	"""Make a repeatable family-grouped split; related families never cross splits."""
	if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
		raise ValueError("split fractions must be between zero and one")
	if train_fraction + validation_fraction >= 1:
		raise ValueError("train and validation fractions must leave test data")
	if len(tasks) < 5:
		raise ValueError("at least five tasks are required for non-empty splits")
	families: dict[str, list[RepairTask]] = {}
	for task in tasks:
		families.setdefault(task.family, []).append(task)
	groups = sorted(families)
	if len(groups) < 3:
		raise ValueError("at least three independent task families are required")
	random.Random(seed).shuffle(groups)
	train_count = max(1, int(len(groups) * train_fraction))
	validation_count = max(1, int(len(groups) * validation_fraction))
	if train_count + validation_count >= len(groups):
		validation_count = len(groups) - train_count - 1
	assignment: dict[str, SplitName] = {}
	for family in groups[:train_count]:
		assignment[family] = "train"
	for family in groups[train_count:train_count + validation_count]:
		assignment[family] = "validation"
	for family in groups[train_count + validation_count:]:
		assignment[family] = "test"
	result: dict[SplitName, tuple[RepairTask, ...]] = {
		name: tuple(task for task in tasks if assignment[task.family] == name)
		for name in ("train", "validation", "test")
	}
	if any(not result[name] for name in result):
		raise ValueError("split produced an empty partition")
	return result


def split_manifest(
	tasks: tuple[RepairTask, ...] = TASKS,
	seed: int = 2026,
	train_fraction: float = 0.5,
	validation_fraction: float = 0.25,
) -> dict[str, object]:
	splits = split_tasks(tasks, seed, train_fraction, validation_fraction)
	task_to_split = {
		task.task_id: split_name
		for split_name, partition in splits.items()
		for task in partition
	}
	return {
		"seed": seed,
		"dataset_sha256": dataset_fingerprint(tasks),
		"counts": {name: len(partition) for name, partition in splits.items()},
		"task_to_split": task_to_split,
		"family_to_split": {
			task.family: task_to_split[task.task_id] for task in tasks
		},
	}


def get_split(name: str, tasks: tuple[RepairTask, ...] = TASKS, seed: int = 2026) -> tuple[RepairTask, ...]:
	if name not in {"train", "validation", "test"}:
		raise ValueError(f"unknown split {name!r}; expected train, validation, or test")
	return split_tasks(tasks, seed)[name]  # type: ignore[index]


def assert_validation_only(split_name: str) -> None:
	"""Fail closed when a tuning operation is pointed at anything but validation."""
	if split_name != "validation":
		raise PermissionError("hyperparameter search may use validation data only")


def validate_behavior(tasks: tuple[RepairTask, ...] = TASKS, timeout_seconds: float = 5.0) -> list[str]:
	"""Check that each reference passes and each intentionally buggy seed is caught."""
	from sandbox import run_candidate

	problems = validate_task_records(tasks)
	if problems:
		return problems
	for task in tasks:
		reference = run_candidate(task.reference_code, task.tests_code, timeout_seconds)
		if not reference.tests_passed:
			problems.append(f"{task.task_id}: reference failed its own tests ({reference.failure_kind})")
		seed = run_candidate(task.starter_code, task.tests_code, timeout_seconds)
		if seed.tests_passed:
			problems.append(f"{task.task_id}: tests do not detect the known starter defect")
	return problems


def require_quality(tasks: tuple[RepairTask, ...] = TASKS, run_behavior: bool = False) -> None:
	problems = validate_behavior(tasks) if run_behavior else validate_task_records(tasks)
	if problems:
		raise DatasetQualityError("Dataset quality checks failed:\n- " + "\n- ".join(problems))