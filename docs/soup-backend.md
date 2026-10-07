# Soup backend (layer streaming)

`spec.backend: soup` trains with [Soup CLI](https://github.com/MakazhanAlpamys/Soup)'s exact layer streaming:
the frozen base model lives in system RAM (or on disk) and one decoder layer at a time is copied to the GPU, so a
model larger than VRAM can be fine-tuned with LoRA. The default backend (`native`) keeps the whole model on the
GPU and is still the right choice when the model fits.

## What was measured here (one RTX 3050 Ti Laptop 4 GB, Windows 11, 16 GB RAM, Soup 0.75.2)

| Model | Steps / seq len | Throughput | GPU memory | Note |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct NF4 | 60 / short (~90 tok) | ~65 tok/s | Soup readout 1.1 GB | short sequences: per-step streaming cost dominates |
| Qwen2.5-3B-Instruct NF4 | 51 / ~360 tok | ~323 tok/s | Soup readout 1.5 GB | |
| Llama-3.1-8B-Instruct NF4 | 51 / ~315 tok | ~176 tok/s | `nvidia-smi` 3.48 GB mid-run (Soup readout 2.0 GB) | free RAM fell to 0.5 GB |

Single short runs on a repacked SQL dataset, not controlled benchmarks. Soup's own claim for 8B on this class of
card is 119.6 tok/s. Adapter quality is checked separately (see `benchmarks/`).

## Setup

1. Install Soup in **its own virtual environment** (never into the tinyforge venv):
   `python -m venv soup-venv && soup-venv/bin/pip install "soup-cli[train]"` (CUDA build of torch first).
2. `export TINYFORGE_SOUP_BIN=/path/to/soup-venv/bin/soup` (Windows: `...\Scripts\soup.exe`).
3. Optional: `TINYFORGE_MODEL_DIR` sets where base models are stored as plain folders (default
   `~/.cache/tinyforge/models`). Soup on Windows rejects the Hugging Face cache layout, so the worker downloads
   the model into a plain folder first.
4. Write a spec with `backend: soup` (see `spec/examples/soup-8b-streaming.yaml`) and run
   `tinyforge worker run --spec job.yaml --out out/`.

## How it is contained

- Soup is never imported; it runs as a subprocess with `--no-telemetry --no-audit-log`, `SOUP_TELEMETRY=0`,
  `HF_HUB_OFFLINE=1`, and `WANDB_DISABLED=true`. (In 0.75.2 telemetry is opt-in and off by default; we force it
  off anyway.)
- Only an allow-list of files is copied from Soup's output into `out/best/` (adapter safetensors and config,
  tokenizer files). `training_args.bin` and other pickles are never copied or loaded.
- Step counts are exact: the worker writes `maxSteps x batchSize x gradAccum` rows (cycling the dataset if
  short) and sets one epoch.
- Soup's progress lines become worker-contract events (`started`, `step`, `artifact` with sha256, `finished`,
  `failed`); exit 4 on out of memory, 75 on preemption.

## Limits (read before relying on it)

- **BETA upstream**, single maintainer; the project itself says its 4 GB numbers predate a correctness fix.
  Pin the Soup version you tested.
- **RAM is the real limit.** An 8B NF4 base needs ~3.6 GB of pinned RAM plus headroom; with the desktop, Docker
  or WSL running it can run out. `tinyforge memory --params-b 8` shows the plan.
- **No resume**: preemption kills the child and loses progress since Soup's last checkpoint.
- **No per-step eval**: `val_loss` gates fail closed on this backend. `gain_pct`/task gates load the model
  resident for evaluation, which only works if it fits in VRAM.
- Single GPU, single node, `method: lora|qlora`, text SFT only.
- Windows paths are exercised on the author's laptop only; Linux has not been run.
