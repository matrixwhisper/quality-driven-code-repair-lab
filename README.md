# Quality-Driven Code Repair Lab

<p align="center">
	<a href="https://www.python.org/"><img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white"></a>
	<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-LLM%20training-EE4C2C?logo=pytorch&logoColor=white"></a>
	<a href="https://huggingface.co/docs/transformers"><img alt="Hugging Face Transformers" src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black"></a>
	<a href="https://huggingface.co/docs/peft"><img alt="PEFT and QLoRA" src="https://img.shields.io/badge/PEFT-QLoRA-8A2BE2?logo=huggingface&logoColor=white"></a>
	<a href="https://pytest.org/"><img alt="pytest" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white"></a>
	<a href="https://github.com/YOUR-GITHUB-USERNAME/quality-driven-code-repair-lab/actions/workflows/ci.yml"><img alt="GitHub Actions status" src="https://github.com/YOUR-GITHUB-USERNAME/quality-driven-code-repair-lab/actions/workflows/ci.yml/badge.svg?branch=main"></a>
</p>

An end-to-end Python experiment for measuring whether supervised fine-tuning
helps an open-source coding model repair small Python bugs. The benchmark puts
dataset review, split discipline, executable feedback, reproducibility, and
failure analysis alongside the model-training code.

> **Experiment status:** the pipeline and CPU tests are implemented, but model
> downloads, QLoRA training, validation search, and held-out model evaluation
> have not been run here. There are no measured model results yet. Do not read
> missing results as zero or as a successful experiment.

## Repository name

Recommended GitHub repository slug: **`quality-driven-code-repair-lab`**.
It is descriptive, readable, and matches the project title. A shorter option
is `code-repair-qlora-benchmark`.

The GitHub Actions badge above uses `YOUR-GITHUB-USERNAME` as a placeholder.
After creating the repository, replace both occurrences with your GitHub
username or organization. The status badge will show a result after the first
workflow run; the technology badges link directly to their official project
pages.

## What the project does

1. Checks a reviewed collection of repair tasks, references, runnable tests,
	 coverage tags, and deterministic split assignments.
2. Evaluates the open-source base model on validation tasks using the shared
	 prompt, decoding, retry, and execution path.
3. Fine-tunes that model on training tasks only with 4-bit NF4 QLoRA, then saves
	 a PEFT adapter and a manifest of the exact training inputs and model revision.
4. Loads the saved adapter for inference and searches repair-agent settings on
	 validation data only.
5. Opens the held-out test split for one final base-versus-adapter comparison
	 after checking the dataset fingerprint, split IDs, model revision, adapter
	 manifest, and validation-search artifact.

The fine-tuning is real: the `train` stage calls Hugging Face Transformers and
PEFT to optimize LoRA parameters on a quantized base model. It is not a mock
trainer or a simulated adapter. The adapter is then loaded through PEFT for
inference. CPU CI does not run this stage.

## Technology

| Area | Implementation |
| --- | --- |
| Language and tests | Python 3.11, `unittest` task suites, `pytest` project tests |
| Coding model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` via Hugging Face Transformers |
| Fine-tuning | PEFT LoRA on 4-bit NF4 weights, double quantization, gradient checkpointing |
| Repair loop | Generate a replacement module, run task tests, feed bounded diagnostics into retries |
| Candidate execution | Disposable Python subprocess, wall-clock timeout, capped output; POSIX resource limits where available |
| Experiment automation | GitHub Actions CPU checks on pushes and pull requests |

Model card: [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct).
Check the current model card and license before redistribution or external use.

## Dataset and quality review

The current corpus is a deliberately small pilot: **11 original, hand-authored
tasks across 10 task families**. Each accepted task stores its specification,
buggy starter, reference implementation, runnable tests, difficulty, coverage
tags, provenance, and review note in `dataset.py`. The review ledger records a
revised example and a rejected example with reasons.

Quality checks flag common vague wording and a few explicit contradictions,
require coverage tags to correspond to named test methods, compile task code,
and verify that references pass while known-buggy starters are caught. These
checks catch obvious defects; they cannot prove that a specification is
unambiguous or that tests cover every intended behavior. Human review is still
required before adding benchmark tasks.

Tasks are grouped by family before splitting so examples from the same family
cannot cross train, validation, and test. With the default seed and split
fractions, the small corpus creates small partitions; one task can therefore
move the score noticeably. This pilot is useful for demonstrating and testing
the experiment pipeline, **not** for broad claims about repair ability or
statistical significance. Family separation within a hand-authored corpus also
does not establish generalization to unseen repositories.

The source examples are original to this project; there are no third-party
benchmark tasks in the current corpus. This repository does not yet declare a
project-wide open-source license. Choose and add an appropriate license before
redistributing the project.

## Setup

Use Python 3.11. Model inference needs enough memory for the base model;
QLoRA training needs a supported NVIDIA CUDA GPU and compatible PyTorch,
CUDA, and `bitsandbytes` builds. On Windows, native QLoRA dependency support
can vary; **WSL2 with a supported CUDA setup** is the recommended route if the
native install fails. The repair benchmark code is Python; Docker is not used.

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch does not detect your GPU, install the PyTorch wheel matching your
CUDA environment using the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then re-check the `bitsandbytes` compatibility for that environment. First
model use downloads model files from Hugging Face and requires network access.

## Run the experiment

Start with the inexpensive checks. These commands do not download a model:

```powershell
python -m pytest -q
python main.py quality
```

Then run the stages in order from the repository root:

```powershell
python main.py baseline
python main.py train
python main.py search --adapter-path runs/adapter
python main.py final --adapter-path runs/adapter --search-results runs/search.json
```

- `baseline` runs the base model on validation tasks and pins the resolved
	model revision in `runs/config.json`.
- `train` fine-tunes on train tasks only, then saves the adapter under
	`runs/adapter/`.
- `search` evaluates candidate agent settings on validation tasks only. It
	requires the saved adapter and records the settings and validation trace.
- `final` checks search and adapter provenance, creates a test-access lock,
	then compares base and adapter on exactly the same held-out test tasks.
	It refuses an accidental second test evaluation in the same run directory.

`python main.py all` automates the same sequence, but separate commands are
easier to resume if a download or GPU training step fails. To use a different
output directory, pass `--run-dir path\to\runs` to each command. Keep that
directory between stages. Do not edit the task corpus or split settings after
search; a changed fingerprint invalidates the final-evaluation gate.

## Clone from GitHub

After creating `quality-driven-code-repair-lab` on GitHub and pushing this
project, clone it from another machine with PowerShell:

```powershell
git clone https://github.com/matrixwhisper/quality-driven-code-repair-lab.git
Set-Location quality-driven-code-repair-lab
```



## GitHub Actions

`.github/workflows/ci.yml` runs the project tests and `python main.py quality`
on pushes, pull requests, and manual dispatch. It installs only the test and
process-measurement dependencies; it does not download model weights, train an
adapter, perform benchmark scoring, or claim GPU results. Standard GitHub
hosted runners are not the QLoRA training environment for this project.

## Metrics and artifacts

Evaluation reports include pass rate after retries, first-attempt compilation
and test success, attempts per task, generation latency per attempt, sampled
model-process RSS, sandbox-process memory samples where available, and peak
CUDA allocated memory where available. Per-task JSON preserves each attempt,
failure category, bounded test output, token counts, and feedback size.

Latency is generation time, not total pipeline time. RSS is sampled and may
miss short-lived peaks; unavailable measurements are reported as unavailable,
not zero. Compare results only when model revision, dataset fingerprint,
settings, and environment records are available.

The `runs/` directory is generated by the pipeline and includes:

- `dataset_manifest.json` and `review_ledger.json`: split IDs, fingerprint,
	quality/review records, and known limitations.
- `config.json` and `runtime_manifest.json`: settings, Python/platform,
	installed package versions, and model metadata.
- `adapter/`: actual adapter/tokenizer and `adapter_manifest.json` recording
	the base revision and train task IDs.
- `search.json`: validation-only hill-climbing trace and selected settings.
- Per-model split JSON and Markdown summaries, plus
	`final_comparison.json` and `final_comparison.md` after final evaluation.
- `final_test_started.json`: a guard recording that held-out test access began.

No score is supplied in advance. Report a stage as **not run** until its command
has completed and saved results.

## Execution safety

The runner starts each candidate in a temporary Python subprocess with a wall
timeout, bounded captured output, and isolated Python import mode. POSIX
systems receive CPU, address-space, file-size, and core-dump limits. On Windows,
the runner currently has no equivalent OS-level CPU/memory restrictions. It
does not reliably block network access, filesystem access, or malicious child
processes and is **not a security sandbox**. Use only this project's reviewed
tasks and generated code on a disposable, non-sensitive machine; do not run
hostile submissions with it.
