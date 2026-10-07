# Roadmap

Each step is finished and measured end to end before the next one starts.

| # | Step | Done when | Status |
|---|---|---|---|
| 1 | Run 4-bit QLoRA on a real GPU | `ft train --quant 4bit` completes, beats base on held-out loss, and peak VRAM and tok/s are reported against fp16 (2.4 GB, 830 tok/s) | **Done**: see results below |
| 2 | Task-specific fine-tune demo | On a narrow task (e.g. structured extraction or SQL), the tuned model beats base by 20% or more on a task metric, not just loss | **Done**: see results below |
| 3 | Fast serving export | The merged model converts to GGUF and generates at least 3x faster than the current 10 tok/s, with outputs matching the original | **Done**: see results below |
| 4 | Job queue and auth | The API needs a token, and queued jobs survive a server restart | **Done**: bearer-token auth plus SQLite-backed queue (`tests/test_server_auth.py`, `tests/test_jobstore.py`) |
| 5 | JAX backend | Same `train()` interface, runs under WSL2 or Linux, and loss matches PyTorch within tolerance on a fixed seed | Todo |
| 6 | More Triton kernels | Fused SwiGLU is faster than PyTorch in a benchmark and passes a numerical test | Todo |
| 7 | Desktop shell | Tauri wraps the existing UI and launches the backend | Todo |
| 8 | Memory-aware planning | `tinyforge memory` probes VRAM/RAM/disk, chunked cross-entropy cuts peak memory, a cost model says where weights live | **Done**: chunked CE 3.65 -> 2.95 GB peak, 289 -> 499 tok/s (`benchmarks/`); the cost model is uncalibrated |
| 9 | Train a model larger than VRAM | An 8B model fine-tunes on a 4 GB GPU through the job spec | **Done**: `backend: soup` (layer streaming), Llama-3.1-8B, 200 steps in 9.5 min (`docs/soup-backend.md`); Windows only, BETA upstream |
| 10 | Data pipeline | Documents and tables become checked training data | **Done (v1)**: `data ingest/tabular/pairs/pii-scan`; PDF/Office via the `docs` extra; pair quality from a real teacher is only spot-checked |
| 11 | Serving | OpenAI-compatible API with guardrails, GGUF export and a llama.cpp engine | **Done (v1)**: `/v1/chat/completions`, `export gguf`, `serve --engine llamacpp`; guardrails are pattern-based only |
| 12 | Multi-GPU, DPO/RL, safety classifier, vLLM | Not built (vLLM deliberately on hold) | Todo |
| 13 | Linux, Kubernetes and multi-node hardening | Run and verified on Linux and a 2-GPU box | Todo (deliberately last) |

## Step 1 results: 4-bit QLoRA vs fp16 (RTX 3050 Ti 4 GB, SmolLM2-360M, 150 steps x 16 examples)

| | fp16 LoRA | 4-bit QLoRA (NF4) |
|---|---|---|
| Median throughput | 833 tok/s | 1258 tok/s |
| Peak VRAM | 2.40 GB | 2.95 GB |
| Micro-batch chosen by planner | 5 examples | 7 examples |
| Training time | 11.4 min | 7.8 min |
| Base val loss (full val set) | 1.314 | 1.389 |
| Tuned val loss | 1.262 | 1.321 |
| Gain over its own base | 4.0% | 4.9% |
| Forgetting on general text | -4.2% (improved) | -1.3% (improved) |
| Merged model export | yes (690 MB) | skipped by design |
| Eval gates | passed | passed |

Reading the numbers honestly:
- QLoRA works and passes every gate. It also trained faster here, but not like-for-like: the planner gave it a larger
  micro-batch (7 vs 5), and the weights occupy less memory.
- It does **not** match fp16 quality. Quantising the base costs about 5.7% held-out loss before any training, and the
  adapter recovers only part of it: final loss 1.321 vs 1.262 (4.7% worse than fp16).
- Peak VRAM is higher than fp16 only because the planner spent the saved weight memory on a bigger batch. QLoRA's real
  benefit is fitting models that do not fit in fp16, which a 360M model does not need. It has not been tested on a
  model that needs it (e.g. 1.5B+).
- Practical rule: use fp16 when it fits, 4-bit only when it does not. The planner already does this.

## Step 2 results: text-to-SQL (`b-mc2/sql-create-context`, SmolLM2-360M, fp16 LoRA)

Run: 3,827 train / 173 held-out examples, 240 steps (about 1 epoch), 5.5 min, 2.4 GB peak VRAM.
Held-out loss 0.865 -> 0.187. Task metrics on all 173 held-out examples, greedy decoding:

| | base, raw prompt | base, explicit instruction | tuned |
|---|---|---|---|
| Exact match (normalised) | 0.6% | 22.0% | **51.4%** |
| Valid SQL (checked with SQLite against the schema) | 3.5% | 90.2% | 92.5% |

- **Gain: +29.5 points exact-match (2.3x) over the best base prompt.** Gate was +20 points: passed.
- The baseline matters. My first run compared against a base model that was never told to write SQL (0.6%), which
  would have claimed +50.9 points. The reported gain is against the stronger, instructed baseline.
- Most of the gain is learning the dataset's conventions (which columns to select, literal formats), not SQL syntax:
  valid-SQL rate is similar (90.2% vs 92.5%).
- Remaining tuned errors are mostly value-literal mistakes (e.g. `"april 6"` vs `6`), not broken SQL.
- Caveats: exact match is strict about structure but case-insensitive about string literals (applies equally to all
  columns); with n=173 the 95% interval on the tuned score is about +/-7 points, so the 29.5-point gain is well
  outside noise, but small differences between runs would not be. This is a narrow in-distribution test (same
  dataset), not evidence of general SQL ability. No execution accuracy yet (needs data in the tables).

Reproduce:
```
tinyforge ft data --source b-mc2/sql-create-context --out data/sql --limit 4000 --max-len 384
tinyforge ft train --steps 240 --max-len 384 --data-dir data/sql --run-dir runs/sql
tinyforge ft task-eval --run-dir runs/sql --n 173 --min-gain-pts 20
```

## Step 3 results: GGUF export via llama.cpp b11380 (merged SQL model, RTX 3050 Ti, full offload)

Same 173 held-out prompts, greedy decoding, batch size 1, 96 max new tokens, end-to-end through llama-server.

| | HF fp16 (baseline) | GGUF f16 | GGUF Q8_0 | GGUF Q4_K_M |
|---|---|---|---|---|
| File size | 690 MB | 690 MB | 367 MB | 256 MB |
| Generation speed | 11.0 tok/s | 148 tok/s (13x) | 227 tok/s (21x) | 249 tok/s (23x) |
| Outputs byte-identical to HF | n/a | 97.1% | 87.3% | 65.9% |
| Exact match (task metric) | 52.0% | 52.6% | 50.9% | 46.8% |
| Valid SQL | 93.6% | 93.6% | 93.6% | 91.9% |

- **Gate (>= 3x and matching outputs): passed** by f16 and Q8_0. Use Q8_0 as the default export; f16 when exactness matters.
- The 5 f16 mismatches are single-token flips on near-ties (e.g. `"2.67 million"` vs `"2.67"`), which is expected from
  different fp16 kernels; task accuracy is unchanged.
- Q4_K_M is fastest and smallest but only 66% identical and 5 points lower exact match. With n=173 (+/-7 points) that is
  suggestive, not proven; treat it as a quality/size trade-off, not a free win.
- Caveats: the HF baseline here (52.0%) differs slightly from the step 2 number (51.4%) because step 2 used batched
  left-padded generation; this run is unbatched. Speed is single-stream only (no concurrent load). Tested on one model
  and one task. Windows CUDA build only.
- Reproduce: `convert_hf_to_gguf.py runs/sql/merged`, `llama-quantize`, then `tools/gguf_parity.py`
  (raw numbers in `runs/sql/gguf/parity.json`).

Steps 1-3 extend what already works. Steps 4-7 are separate pieces.
