"""Iterative code repair driven by task-test feedback."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Protocol

from dataset import RepairTask
from models import GenerationResult, strip_code_fence
from sandbox import SandboxResult, run_candidate


class CodeGenerator(Protocol):
	def generate(
		self,
		prompt: str,
		max_new_tokens: int = 384,
		temperature: float = 0.0,
		top_p: float = 1.0,
	) -> GenerationResult: ...


@dataclass(frozen=True)
class AgentSettings:
	max_retries: int = 1
	feedback_chars: int = 1600
	max_new_tokens: int = 384
	temperature: float = 0.0
	top_p: float = 1.0
	sandbox_timeout_seconds: float = 8.0

	def __post_init__(self) -> None:
		if self.max_retries < 0 or self.feedback_chars < 0:
			raise ValueError("retry and feedback limits must be non-negative")
		if self.max_new_tokens < 1 or self.sandbox_timeout_seconds <= 0:
			raise ValueError("generation token limit and sandbox timeout must be positive")


@dataclass(frozen=True)
class AttemptRecord:
	attempt_number: int
	generation_latency_seconds: float
	input_tokens: int
	output_tokens: int
	compile_success: bool
	tests_passed: bool
	failure_kind: str | None
	sandbox_seconds: float
	feedback_sent_chars: int
	process_rss_before_bytes: int | None
	process_rss_after_bytes: int | None
	sandbox_peak_rss_bytes: int | None
	cuda_peak_allocated_bytes: int | None
	diagnostic_output: str
	error: str | None = None


@dataclass(frozen=True)
class RepairOutcome:
	task_id: str
	attempts: tuple[AttemptRecord, ...]
	final_code: str
	passed: bool

	@property
	def attempt_count(self) -> int:
		return len(self.attempts)

	@property
	def total_generation_latency_seconds(self) -> float:
		return sum(attempt.generation_latency_seconds for attempt in self.attempts)

	def to_dict(self) -> dict[str, object]:
		return {
			"task_id": self.task_id,
			"attempts": [asdict(attempt) for attempt in self.attempts],
			"final_code": self.final_code,
			"passed": self.passed,
			"attempt_count": self.attempt_count,
			"total_generation_latency_seconds": self.total_generation_latency_seconds,
		}


def _failure_feedback(result: SandboxResult, limit: int) -> str:
	message = f"Failure kind: {result.failure_kind or 'unknown'}\nTest output:\n{result.output.strip()}"
	if result.failure_kind == "compile_error":
		message = f"Failure kind: compile_error\nPython could not compile the candidate.\n{result.output.strip()}"
	return message[-limit:] if limit else ""


def repair_task(
	task: RepairTask,
	model: CodeGenerator,
	settings: AgentSettings | None = None,
) -> RepairOutcome:
	"""Generate, execute, and optionally retry using only the task's public tests."""
	settings = settings or AgentSettings()
	current_code = task.starter_code
	feedback: str | None = None
	attempts: list[AttemptRecord] = []

	for attempt_number in range(1, settings.max_retries + 2):
		prompt_task = replace(task, starter_code=current_code)
		prompt = prompt_task.prompt(feedback)
		feedback_chars = len(feedback or "")
		try:
			generation = model.generate(
				prompt,
				max_new_tokens=settings.max_new_tokens,
				temperature=settings.temperature,
				top_p=settings.top_p,
			)
			candidate = strip_code_fence(generation.text)
			result = run_candidate(candidate, task.tests_code, settings.sandbox_timeout_seconds)
			attempts.append(AttemptRecord(
				attempt_number=attempt_number,
				generation_latency_seconds=generation.latency_seconds,
				input_tokens=generation.input_tokens,
				output_tokens=generation.output_tokens,
				compile_success=result.compile_success,
				tests_passed=result.tests_passed,
				failure_kind=result.failure_kind,
				sandbox_seconds=result.duration_seconds,
				feedback_sent_chars=feedback_chars,
				process_rss_before_bytes=generation.process_rss_before_bytes,
				process_rss_after_bytes=generation.process_rss_after_bytes,
				sandbox_peak_rss_bytes=result.peak_rss_bytes,
				cuda_peak_allocated_bytes=generation.cuda_peak_allocated_bytes,
				diagnostic_output=result.output,
			))
		except Exception as exc:
			candidate = current_code
			result = SandboxResult(
				compile_success=False,
				tests_passed=False,
				failure_kind="generation_error",
				return_code=None,
				duration_seconds=0.0,
				output=f"{type(exc).__name__}: {exc}",
			)
			attempts.append(AttemptRecord(
				attempt_number=attempt_number,
				generation_latency_seconds=0.0,
				input_tokens=0,
				output_tokens=0,
				compile_success=False,
				tests_passed=False,
				failure_kind="generation_error",
				sandbox_seconds=0.0,
				feedback_sent_chars=feedback_chars,
				process_rss_before_bytes=None,
				process_rss_after_bytes=None,
				sandbox_peak_rss_bytes=None,
				cuda_peak_allocated_bytes=None,
				diagnostic_output=result.output[:16000],
				error=result.output[:1000],
			))
		if result.tests_passed:
			return RepairOutcome(task.task_id, tuple(attempts), candidate, True)
		current_code = candidate or current_code
		feedback = _failure_feedback(result, settings.feedback_chars)

	return RepairOutcome(task.task_id, tuple(attempts), current_code, False)
