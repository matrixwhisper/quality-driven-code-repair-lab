"""Open-source coding-model inference with optional PEFT adapter loading."""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@dataclass(frozen=True)
class GenerationResult:
	text: str
	latency_seconds: float
	input_tokens: int
	output_tokens: int
	process_rss_before_bytes: int | None
	process_rss_after_bytes: int | None
	cuda_peak_allocated_bytes: int | None

	def to_dict(self) -> dict[str, object]:
		return asdict(self)


def strip_code_fence(text: str) -> str:
	"""Remove one outer markdown fence without rewriting generated source."""
	value = text.strip()
	match = re.fullmatch(r"```(?:python|py)?\s*\n?(.*?)\n?```", value, flags=re.DOTALL | re.IGNORECASE)
	return match.group(1).strip() + "\n" if match else value + ("\n" if value else "")


def _rss_bytes() -> int | None:
	try:
		import psutil
		info = psutil.Process(os.getpid()).memory_info()
		return int(getattr(info, "peak_wset", info.rss))
	except (ImportError, OSError):
		return None


class HFCodeModel:
	"""A single shared generation interface for base and adapter evaluations."""

	def __init__(
		self,
		model_id: str = DEFAULT_MODEL_ID,
		revision: str = "main",
		adapter_path: str | None = None,
		trust_remote_code: bool = False,
	) -> None:
		try:
			import torch
			from transformers import AutoModelForCausalLM, AutoTokenizer
		except ImportError as exc:
			raise RuntimeError(
				"Model inference dependencies are missing; install requirements.txt and a compatible PyTorch build."
			) from exc

		self.torch = torch
		self.model_id = model_id
		self.adapter_path = adapter_path
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_id, revision=revision, trust_remote_code=trust_remote_code,
		)
		if self.tokenizer.pad_token_id is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
		use_cuda = torch.cuda.is_available()
		dtype = (
			torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
			else torch.float16 if use_cuda else torch.float32
		)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_id,
			revision=revision,
			torch_dtype=dtype,
			device_map="auto" if use_cuda else None,
			trust_remote_code=trust_remote_code,
		)
		if not use_cuda:
			self.model.to("cpu")
		if adapter_path:
			try:
				from peft import PeftModel
			except ImportError as exc:
				raise RuntimeError("PEFT is required to load a saved QLoRA adapter") from exc
			self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
		self.model.eval()
		self.revision = getattr(self.model.config, "_commit_hash", None) or revision

	def generate(
		self,
		prompt: str,
		max_new_tokens: int = 384,
		temperature: float = 0.0,
		top_p: float = 1.0,
	) -> GenerationResult:
		if max_new_tokens < 1:
			raise ValueError("max_new_tokens must be positive")
		if temperature < 0 or not 0 < top_p <= 1:
			raise ValueError("temperature must be non-negative and top_p in (0, 1]")
		torch = self.torch
		if getattr(self.tokenizer, "chat_template", None):
			input_ids = self.tokenizer.apply_chat_template(
				[{"role": "user", "content": prompt}],
				tokenize=True,
				add_generation_prompt=True,
				return_tensors="pt",
				truncation=True,
				max_length=4096,
			)
			if isinstance(input_ids, dict):
				encoded = input_ids
			else:
				encoded = {"input_ids": input_ids}
				encoded["attention_mask"] = torch.ones_like(input_ids)
		else:
			encoded = self.tokenizer(
				prompt, return_tensors="pt", truncation=True, max_length=4096,
			)
		input_device = self.model.get_input_embeddings().weight.device
		encoded = {key: value.to(input_device) for key, value in encoded.items()}
		input_tokens = int(encoded["input_ids"].shape[-1])
		generation_args: dict[str, Any] = {
			"max_new_tokens": max_new_tokens,
			"do_sample": temperature > 0,
			"pad_token_id": self.tokenizer.pad_token_id,
			"eos_token_id": self.tokenizer.eos_token_id,
			"use_cache": True,
		}
		if temperature > 0:
			generation_args.update({"temperature": temperature, "top_p": top_p})
		if torch.cuda.is_available():
			for device_index in range(torch.cuda.device_count()):
				torch.cuda.reset_peak_memory_stats(device_index)
		rss_before = _rss_bytes()
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		started = time.perf_counter()
		with torch.inference_mode():
			output = self.model.generate(**encoded, **generation_args)
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		latency = time.perf_counter() - started
		generated_ids = output[0, input_tokens:]
		text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
		peak_gpu = None
		if torch.cuda.is_available():
			peak_gpu = max(int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count()))
		return GenerationResult(
			text=text,
			latency_seconds=latency,
			input_tokens=input_tokens,
			output_tokens=int(generated_ids.numel()),
			process_rss_before_bytes=rss_before,
			process_rss_after_bytes=_rss_bytes(),
			cuda_peak_allocated_bytes=peak_gpu,
		)

	def metadata(self) -> dict[str, object]:
		torch = self.torch
		return {
			"model_id": self.model_id,
			"resolved_revision": self.revision,
			"adapter_path": self.adapter_path,
			"device": str(self.model.get_input_embeddings().weight.device),
			"cuda_available": bool(torch.cuda.is_available()),
			"cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
		}
