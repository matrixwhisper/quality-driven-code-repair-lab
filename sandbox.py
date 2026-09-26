"""Disposable-process execution for benchmark candidates.

This is a resource-bounded test harness, not a security sandbox. Never run
untrusted submissions on a sensitive host; use an OS/container boundary for
hostile code. Docker is intentionally not required by this project.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


MAX_SOURCE_BYTES = 256_000
MAX_OUTPUT_BYTES = 16_000


@dataclass(frozen=True)
class SandboxResult:
	compile_success: bool
	tests_passed: bool
	failure_kind: str | None
	return_code: int | None
	duration_seconds: float
	output: str
	peak_rss_bytes: int | None = None

	def to_dict(self) -> dict[str, object]:
		return asdict(self)


def _read_capped(stream, output: bytearray) -> None:
	while True:
		chunk = stream.read(4096)
		if not chunk:
			return
		output.extend(chunk)
		if len(output) > MAX_OUTPUT_BYTES:
			del output[:-MAX_OUTPUT_BYTES]


def _posix_limits(timeout_seconds: float, memory_limit_mb: int):
	def apply_limits() -> None:
		import resource

		cpu_limit = max(2, int(timeout_seconds) + 1)
		resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
		resource.setrlimit(resource.RLIMIT_AS, (memory_limit_mb * 1024**2, memory_limit_mb * 1024**2))
		resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
		resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

	return apply_limits


def _classify_test_failure(return_code: int, output: str) -> str:
	if "AssertionError" in output or "FAIL:" in output:
		return "assertion_failure"
	if "ModuleNotFoundError" in output or "ImportError" in output:
		return "import_failure"
	for exception_name in (
		"TypeError", "NameError", "IndexError", "KeyError", "ValueError",
		"AttributeError", "ZeroDivisionError", "OverflowError", "RuntimeError",
	):
		if exception_name in output:
			return f"runtime_{exception_name.lower()}"
	return "test_failure" if return_code else "unknown_failure"


def run_candidate(
	candidate_code: str,
	tests_code: str,
	timeout_seconds: float = 8.0,
	memory_limit_mb: int = 1024,
) -> SandboxResult:
	"""Compile then run unittest in a temporary child process with a hard timeout."""
	started = time.perf_counter()
	if timeout_seconds <= 0 or timeout_seconds > 300:
		raise ValueError("timeout_seconds must be in (0, 300]")
	if memory_limit_mb < 128:
		raise ValueError("memory_limit_mb must be at least 128")
	if len(candidate_code.encode("utf-8")) > MAX_SOURCE_BYTES:
		return SandboxResult(False, False, "source_too_large", None, 0.0, "Candidate exceeded source-size limit")
	if len(tests_code.encode("utf-8")) > MAX_SOURCE_BYTES:
		return SandboxResult(False, False, "tests_too_large", None, 0.0, "Test suite exceeded source-size limit")
	try:
		compile(candidate_code, "candidate.py", "exec")
		compile(tests_code, "tests.py", "exec")
	except SyntaxError as exc:
		return SandboxResult(False, False, "compile_error", None, time.perf_counter() - started, f"{exc.__class__.__name__}: {exc}")

	with tempfile.TemporaryDirectory(prefix="repair_task_") as temporary_directory:
		workspace = Path(temporary_directory)
		solution_path = workspace / "solution.py"
		tests_path = workspace / "tests.py"
		solution_path.write_text(candidate_code, encoding="utf-8")
		tests_path.write_text(tests_code, encoding="utf-8")
		bootstrap = (
			"import runpy, sys; "
			"workspace, test_path = sys.argv[1:3]; "
			"sys.path.insert(0, workspace); "
			"sys.argv[:] = [test_path]; "
			"runpy.run_path(test_path, run_name='__main__')"
		)
		command = [sys.executable, "-I", "-c", bootstrap, str(workspace), str(tests_path)]
		environment = {
			key: os.environ[key]
			for key in ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL")
			if key in os.environ
		}
		environment.setdefault("PATH", os.defpath)
		environment.update({
			"PYTHONNOUSERSITE": "1",
			"PYTHONDONTWRITEBYTECODE": "1",
			"HOME": str(workspace),
			"USERPROFILE": str(workspace),
			"TMP": str(workspace),
			"TEMP": str(workspace),
		})
		output = bytearray()
		try:
			process = subprocess.Popen(
				command,
				cwd=workspace,
				env=environment,
				stdin=subprocess.DEVNULL,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				start_new_session=(os.name == "posix"),
				preexec_fn=_posix_limits(timeout_seconds, memory_limit_mb) if os.name == "posix" else None,
			)
			reader = threading.Thread(target=_read_capped, args=(process.stdout, output), daemon=True)
			reader.start()
			deadline = time.monotonic() + timeout_seconds
			peak_rss_bytes: int | None = None
			try:
				import psutil
				child = psutil.Process(process.pid)
			except (ImportError, OSError):
				child = None
			while process.poll() is None:
				if child is not None:
					try:
						peak_rss_bytes = max(peak_rss_bytes or 0, child.memory_info().rss)
					except (OSError, psutil.Error):
						child = None
				if time.monotonic() >= deadline:
					if os.name == "posix":
						try:
							os.killpg(process.pid, signal.SIGKILL)
						except ProcessLookupError:
							pass
					else:
						process.kill()
					process.wait()
					reader.join(timeout=1)
					return SandboxResult(
						True, False, "timeout", process.returncode,
						time.perf_counter() - started, output.decode("utf-8", errors="replace"), peak_rss_bytes,
					)
				time.sleep(0.025)
			reader.join(timeout=1)
			text = output.decode("utf-8", errors="replace")
			test_count = re.search(r"Ran\s+(\d+)\s+tests?\b", text)
			if process.returncode == 0 and (test_count is None or int(test_count.group(1)) == 0):
				return SandboxResult(
					True, False, "no_tests_discovered", process.returncode,
					time.perf_counter() - started, text, peak_rss_bytes,
				)
			return SandboxResult(
				True,
				process.returncode == 0,
				None if process.returncode == 0 else _classify_test_failure(process.returncode, text),
				process.returncode,
				time.perf_counter() - started,
				text,
				peak_rss_bytes,
			)
		except OSError as exc:
			return SandboxResult(
				True, False, "runner_error", None, time.perf_counter() - started, str(exc),
			)
