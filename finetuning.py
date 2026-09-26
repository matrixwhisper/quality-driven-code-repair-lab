"""CUDA QLoRA fine-tuning for the benchmark's training split only."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config import ExperimentConfig
from dataset import RepairTask, dataset_fingerprint
from models import DEFAULT_MODEL_ID


class _RepairSFTDataset:
	def __init__(self, rows: list[dict[str, list[int]]]) -> None:
		import torch
		self.torch = torch
		self.rows = rows

	def __len__(self) -> int:
		return len(self.rows)

	def __getitem__(self, index: int) -> dict[str, Any]:
		row = self.rows[index]
		return {key: self.torch.tensor(value, dtype=self.torch.long) for key, value in row.items()}


def _training_row(tokenizer, task: RepairTask, max_length: int) -> dict[str, list[int]]:
	user_message = {"role": "user", "content": task.prompt()}
	assistant_message = {"role": "assistant", "content": task.reference_code}
	prompt_ids = tokenizer.apply_chat_template(
		[user_message], tokenize=True, add_generation_prompt=True,
	)
	full_ids = tokenizer.apply_chat_template(
		[user_message, assistant_message], tokenize=True, add_generation_prompt=False,
	)
	if not full_ids[:len(prompt_ids)] == prompt_ids:
		raise ValueError(f"chat template does not preserve assistant prefix for task {task.task_id}")
	if len(prompt_ids) >= max_length:
		raise ValueError(f"prompt leaves no target tokens for task {task.task_id}")
	input_ids = full_ids[:max_length]
	labels = [-100] * min(len(prompt_ids), len(input_ids))
	labels.extend(input_ids[len(labels):])
	return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def _collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
	import torch
	longest = max(len(row["input_ids"]) for row in batch)
	input_ids, attention_masks, labels = [], [], []
	for row in batch:
		padding = longest - len(row["input_ids"])
		input_ids.append(row["input_ids"] + [pad_token_id] * padding)
		attention_masks.append(row["attention_mask"] + [0] * padding)
		labels.append(row["labels"] + [-100] * padding)
	return {
		"input_ids": torch.tensor(input_ids, dtype=torch.long),
		"attention_mask": torch.tensor(attention_masks, dtype=torch.long),
		"labels": torch.tensor(labels, dtype=torch.long),
	}


def train_qlora(
	config: ExperimentConfig,
	train_tasks: tuple[RepairTask, ...],
	output_dir: str | Path,
) -> Path:
	"""Train and save a real PEFT adapter; refuses CPU and empty training data."""
	if not train_tasks:
		raise ValueError("training split is empty")
	if any(task.review_status != "accepted" for task in train_tasks):
		raise ValueError("training accepts only reviewed, accepted tasks")
	task_ids = [task.task_id for task in train_tasks]
	if len(task_ids) != len(set(task_ids)):
		raise ValueError("duplicate task ids in training input")
	try:
		import torch
		from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
		from transformers import (
			AutoModelForCausalLM,
			AutoTokenizer,
			BitsAndBytesConfig,
			Trainer,
			TrainingArguments,
		)
	except ImportError as exc:
		raise RuntimeError("Install the CUDA-compatible requirements to train QLoRA") from exc
	if not torch.cuda.is_available():
		raise RuntimeError("QLoRA requires a CUDA-capable GPU; CPU training is intentionally disabled")
	if not torch.cuda.is_bf16_supported():
		compute_dtype = torch.float16
		use_bf16 = False
	else:
		compute_dtype = torch.bfloat16
		use_bf16 = True

	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed_all(config.seed)
	quantization = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=compute_dtype,
	)
	tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained(
		config.model_id,
		revision=config.revision,
		quantization_config=quantization,
		device_map={"": torch.cuda.current_device()},
	)
	resolved_revision = getattr(model.config, "_commit_hash", None) or config.revision
	model.config.use_cache = False
	model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
	adapter_config = LoraConfig(
		r=config.qlora_rank,
		lora_alpha=config.qlora_alpha,
		lora_dropout=config.qlora_dropout,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules="all-linear",
	)
	model = get_peft_model(model, adapter_config)
	rows = [_training_row(tokenizer, task, config.max_sequence_length) for task in train_tasks]
	dataset = _RepairSFTDataset(rows)
	destination = Path(output_dir)
	destination.mkdir(parents=True, exist_ok=True)
	training_args = TrainingArguments(
		output_dir=str(destination / "trainer_state"),
		num_train_epochs=config.train_epochs,
		learning_rate=config.learning_rate,
		per_device_train_batch_size=1,
		gradient_accumulation_steps=config.gradient_accumulation_steps,
		logging_steps=1,
		save_strategy="no",
		report_to=[],
		seed=config.seed,
		data_seed=config.seed,
		fp16=not use_bf16,
		bf16=use_bf16,
		gradient_checkpointing=True,
		remove_unused_columns=False,
		optim="paged_adamw_8bit",
	)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
		data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id),
	)
	train_result = trainer.train()
	trainer.save_model(str(destination))
	tokenizer.save_pretrained(str(destination))
	manifest = {
		"status": "completed",
		"method": "QLoRA",
		"quantization": "4-bit NF4 with double quantization",
		"base_model_id": config.model_id or DEFAULT_MODEL_ID,
		"base_model_revision": resolved_revision,
		"adapter_path": str(destination.resolve()),
		"dataset_sha256": dataset_fingerprint(),
		"train_task_ids": task_ids,
		"split_used": "train",
		"validation_used_for_gradient_updates": False,
		"test_accessed": False,
		"seed": config.seed,
		"qlora": {
			"rank": config.qlora_rank,
			"alpha": config.qlora_alpha,
			"dropout": config.qlora_dropout,
			"target_modules": "all-linear",
		},
		"training_config": asdict(config) | {"run_dir": str(config.run_dir)},
		"train_metrics": train_result.metrics,
		"torch_version": torch.__version__,
		"gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
	}
	(destination / "adapter_manifest.json").write_text(
		json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
		encoding="utf-8",
	)
	return destination
