"""Validated experiment settings and seed helpers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
	model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
	revision: str = "main"
	run_dir: Path = Path("runs")
	seed: int = 2026
	train_fraction: float = 0.5
	validation_fraction: float = 0.25
	max_new_tokens: int = 384
	temperature: float = 0.0
	top_p: float = 1.0
	max_retries: int = 1
	feedback_chars: int = 1600
	sandbox_timeout_seconds: float = 8.0
	qlora_rank: int = 16
	qlora_alpha: int = 32
	qlora_dropout: float = 0.05
	train_epochs: float = 2.0
	learning_rate: float = 2e-4
	max_sequence_length: int = 1536
	gradient_accumulation_steps: int = 8
	search_evaluation_budget: int = 5
	extra: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not self.model_id.strip():
			raise ValueError("model_id must not be empty")
		if not 0 < self.train_fraction < 1:
			raise ValueError("train_fraction must be between zero and one")
		if not 0 < self.validation_fraction < 1:
			raise ValueError("validation_fraction must be between zero and one")
		if self.train_fraction + self.validation_fraction >= 1:
			raise ValueError("split fractions must leave a non-empty test split")
		if self.max_new_tokens < 1 or self.max_retries < 0:
			raise ValueError("max_new_tokens must be positive and max_retries non-negative")
		if self.feedback_chars < 0 or self.sandbox_timeout_seconds <= 0:
			raise ValueError("feedback_chars must be non-negative and timeout positive")
		if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
			raise ValueError("temperature must be in [0, 2] and top_p in (0, 1]")
		if self.qlora_rank < 1 or self.qlora_alpha < 1 or not 0 <= self.qlora_dropout < 1:
			raise ValueError("invalid LoRA rank, alpha, or dropout")
		if self.train_epochs <= 0 or self.learning_rate <= 0:
			raise ValueError("training epochs and learning rate must be positive")
		if self.max_sequence_length < 128 or self.gradient_accumulation_steps < 1:
			raise ValueError("sequence length must be >= 128 and accumulation >= 1")
		if self.search_evaluation_budget < 1:
			raise ValueError("search_evaluation_budget must be positive")

	def to_dict(self) -> dict[str, Any]:
		result = asdict(self)
		result["run_dir"] = str(self.run_dir)
		return result

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
		values = dict(data)
		if "run_dir" in values:
			values["run_dir"] = Path(values["run_dir"])
		return cls(**values)

	def save(self, path: str | Path) -> Path:
		destination = Path(path)
		destination.parent.mkdir(parents=True, exist_ok=True)
		destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
		return destination

	@classmethod
	def load(cls, path: str | Path) -> "ExperimentConfig":
		return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def seed_everything(seed: int) -> None:
	random.seed(seed)
	try:
		import numpy as np
		np.random.seed(seed)
	except ImportError:
		pass
	try:
		import torch
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
		if hasattr(torch.backends, "cudnn"):
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
	except ImportError:
		pass
