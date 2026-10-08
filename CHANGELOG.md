# Changelog
Format: Keep a Changelog (https://keepachangelog.com), versioning: SemVer. Pre-1.0: minor versions may break.

## [Unreleased]

## [0.1.2] - 2026-10-09
### Fixed
- README status line (shown on PyPI) wrongly said multi-GPU was untested; it now states the Kaggle 2x T4 result.

## [0.1.1] - 2026-10-08
### Added
- Multi-GPU: data-parallel LoRA fine-tuning on one machine (`tinyforge ft train --gpus N`, or `resources.gpus: N`
  in a job spec). Each GPU trains on its own full batch (effective batch N x; `--split-batch` splits one batch instead) and the LoRA gradients are averaged each step; the run
  proves the replicas stayed identical. Tested with real 2-process runs (CPU, gloo) on Windows and Linux and on a
  Kaggle 2x T4 machine (`notebooks/kaggle_multi_gpu.ipynb`): NCCL works, replicas stay identical, resume works.
  Measured throughput is 1.2x-1.4x of one GPU on a small 360M job (not 2x); cause not yet established.
- `docs/install.md`: Linux, Windows and Colab/Kaggle instructions side by side, with what is tested where.
- Colab and Kaggle test notebooks (`notebooks/`).
- PyPI classifiers for Linux and Python 3.13.

### Fixed (found by running the published 0.1.0 on a Linux Colab T4 and a Kaggle 2x T4)
- `tinyforge generate` / `eval` looked in `runs/default` while `pipeline` wrote `runs/<preset>`; they now find the
  newest checkpoint and say what to do when there is none.
- Colab/Kaggle's preinstalled `torchao` (0.10) made `peft` refuse to start; tinyforge now hides a too-old torchao
  from its own process (opt out: `TINYFORGE_KEEP_TORCHAO=1`) and explains other version clashes in plain words.
- A broken TensorFlow (Colab ships one; llama.cpp's requirements downgrade `protobuf` under it) crashed every
  command on import: tinyforge now turns off TensorFlow/Flax/JAX in `transformers` (it is PyTorch-only).
- GPUs without native bf16 (compute capability below 8, e.g. T4) were treated as bf16-capable and trained on slow
  emulation; they now use fp16 with loss scaling.
- Held-out loss could come out NaN on fp16 GPUs (a bf16-trained model overflowing during evaluation while training
  stayed finite); an overflowing batch is now recomputed in bf16 and reported as diagnostic FT007.
- `bench run --hf-model` passed `load_in_4bit` to newer transformers, which rejects it; it now loads fp16 and
  `--four-bit` is opt-in.

## [0.1.0] - 2026-10-08
### Added
- Optional Soup training backend (`spec.backend: soup`): layer streaming trains models larger than VRAM (8B on
  a 4 GB GPU measured); isolated subprocess, telemetry off, allow-listed outputs. See docs/soup-backend.md.
- `tinyforge memory` (VRAM/RAM/disk probe and weight placement plan), chunked cross-entropy, memory-tier cost model.
- Data pipeline: `data ingest` (extract/clean/chunk/dedupe, PDF/Office via the `docs` extra), `data tabular`
  (execution-verified text-to-SQL), `data pairs` (split-before-generation, grounded teacher Q&A), `data pii-scan`
  and `--pii flag|redact|drop`.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` (JSON + SSE) with guardrails v1 (secret block/redact,
  optional PII policy and denylist, counters on /metrics).
- `tinyforge bench list|suggest|run`: benchmark catalog with a hardware-aware recommender (decode-speed model from
  memory bandwidth, time budget, what cannot run and why), `lm-evaluation-harness` runner and a built-in SQL
  execution-accuracy check. The harness itself has not been run end to end yet.
- `tinyforge export gguf` and `tinyforge serve --engine llamacpp` (managed llama-server behind the same API, auto
  GPU layers, slots with a short queue). See docs/llamacpp-engine.md.
- Host RAM/swap gauges on /metrics and a training Grafana dashboard.
- API bearer auth, SQLite job queue, /metrics, TLS/CORS/rate limiting.
- `forgectl` (validate, plan, doctor, run, agent, net check); JobSpec v1alpha1; Helm chart and CRD; Terraform.
- Supply chain: SHA-pinned Actions, govulncheck, gitleaks, Dependabot, hashed dependency locks.

### Fixed
- `ft eval` no longer passes a model whose base is untrained or whose output is degenerate (FE010/FE011); relative-gain
  gates fail closed in that case; a model without a chat template gets FD007 instead of a traceback.
- `transformers` 5.19.0 is excluded (it fails to import on CPU-only PyTorch 2.6); hashed locks regenerated.
- The job-spec schema now ships inside the wheel (a pip-installed `worker run` could not find it before).
- Fine-tuning now ends replies with the chat template's turn terminator instead of the tokenizer's EOS (wrong stop
  token for base models such as Qwen2.5 base).
- ETL boilerplate filter no longer drops repeated sentences, table rows or code lines; PDF line wraps are rejoined.
