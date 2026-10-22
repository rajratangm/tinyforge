# Kernel engine: handoff (updated 2026-10-22, end of session)

Read this first when resuming. Branch: `feat/kernel-engine` (local only, **nothing pushed**, not merged to `main`).
User guide: `docs/kernels.md`. Spec: `docs/superpowers/specs/2026-10-22-kernel-engine-design.md`. Plan:
`docs/superpowers/plans/2026-10-22-kernel-engine.md` (Tasks 1-16). Ledger (git-ignored scratch):
`.superpowers/sdd/2026-10-22-kernel-engine/progress.md`.

## Why this exists
The user said tinyforge "does nothing" for kernels and wants dynamic kernels used in real training and inference.
At the start the only kernel was a 27-line forward-only RMSNorm used by the toy GPT, not by the HF fine-tune/serve
path. Direction chosen: a verified Triton kernel library, then auto-fusion, then user-written kernels. Triton only
(no nvcc/MSVC on the laptop). JAX is parked (no native Windows GPU support).

## State: all 16 plan tasks are implemented and tested

| Task | What | Status |
|---|---|---|
| 1 | Triton smoke test, `kernels doctor` | done (`benchmarks/kernels-smoke.json`) |
| 2-4 | registry, verifier (outputs + gradients), autotuner + per-GPU cache | done |
| 5-9 | RMSNorm, SwiGLU, RoPE (GQA), fused cross-entropy, residual+RMSNorm (inference) | done |
| 10 | patcher: `apply_kernels`, `restore_kernels`, automatic fallback | done |
| 11 | `--kernels` on ft/serve/JobSpec/CRD/Go, first-batch drift guard | done |
| 12 | `kernels verify` / `kernels bench`, acceptance runs | done; default left at `off` (see Rulings) |
| 13 | `codegen.py`: Triton fwd+bwd generator for op chains (17 ops, row reductions) | done |
| 14 | `fuse.py`: torch.fx chain finder | done |
| 15 | auto-fusion in the patcher, `--fuse`, `kernels fuse --model` | done |
| 16 | user-defined kernels: `kernels new`, `verify --file`, `--kernel-path` | done |

Re-run before trusting anything: `.venv\Scripts\python -m pytest -q --no-cov`, `.venv\Scripts\ruff check src tests`,
and the Go tests with `GOROOT=tools/go-sdk/go GOPATH=tools/go-sdk/gopath` (`go vet ./... && go test ./...`).
Final run on 2026-10-22: **534 passed, 1 skipped**, ruff clean, `go vet` and `go test ./...` clean.
Review status: there was no independent whole-branch review (it needs a separate reviewer; only a self-review by
the author was done), so a fresh reviewer pass over `git diff main...feat/kernel-engine -- src/tinyforge/kernels`
is still worth doing before merging.

## Measured results (one machine: Windows 11, RTX 3050 Ti 4 GB; all in `benchmarks/`)
* SmolLM2-360M bf16, 30 steps, off vs auto: 1406 -> 1611 tok/s (+14.5%), peak 2.19 -> 1.34 GB (-39%), same loss.
  Two earlier valid runs gave +17.4% and +14.5%.
* Qwen2.5-1.5B 4-bit QLoRA, 20 steps: 421 -> 439 tok/s (+4.3%), peak 2.95 -> 2.50 GB, same loss.
* Generated activation kernels (`tools/bench_fusion.py`): gelu_new 6.8x, gelu_fast 6.5x, quick_gelu 2.7x,
  relu2 2.1x, memory 144/192/96/96 -> 48 MB. mish is not fused (softplus unsupported).
* Greedy generation, kernels on vs off: 4 of 8 prompts identical over 80 tokens, others diverge at token 6-43
  (bf16 rounding). Documented in `docs/kernels.md`; do not claim token-identical output.

## Rulings made on the user's behalf
* Default `--kernels` stays `off`: every result is from one Windows laptop and generation is not bit-identical;
  flip to `auto` after Linux + a second GPU repeat the acceptance runs (change `FTConfig.kernels`, the CLI
  `KernelsOpt` defaults, schema/CRD/Go defaults, docs). Cost if wrong: users must pass `--kernels auto`.
* User kernels load only from explicit `--kernel-path` / `TINYFORGE_KERNEL_PATH`, never from the current directory,
  a JobSpec or the network (they are arbitrary code). The spec originally said `./tinyforge_kernels` implicitly.
* Template for `kernels new` lives in the package (`kernels/templates/geglu.py`) so it ships in the wheel.
* Fused CE is tuned on a proxy width (<= 512) to fit 4 GB; passed to the loss as params, not patched as a module.
* Drift guard bypasses patches with `_Patches.enabled` instead of restoring and re-applying.
* Branch in place rather than a worktree (venv, CUDA and tools live in the main checkout).

## Bugs found by running it for real (each fixed with a failing test first)
1. SwiGLU forward read a loop-shared `params` variable (late binding) and silently fell back (KeyError BLOCK).
2. Gradient checkpointing: torch's `_StopRecomputationError` passed through the patched forward was treated as a
   kernel failure and crashed training. Control-flow exceptions now pass through untouched.
3. RoPE indexed cos/sin per batch but HF passes batch-1 cos/sin during training: silent wrong values.
4. QLoRA (`prepare_model_for_kbit_training` casts every weight to fp32): fused CE backward did bf16 @ fp32.
5. The tuner timed the reference once, on a cold GPU clock, so kernels looked up to 10x faster than they were and
   the cache kept it; the reference is now timed again after the candidates and the faster reading wins, with a
   0.2 s GPU warm-up.
6. Tuner timed inference-only kernels against a forward+backward reference; `kernels bench` could leave a stale
   result file after a crash; test fixtures that restored a registry snapshot deleted built-in kernels.
Lesson: the fallback design hides broken kernels, so tests assert `report.skipped == {}` and the report is rewritten
at the end of training.

## Known gaps and risks (be honest about these)
* Verified only on this laptop (torch 2.6.0+cu124, triton 3.8.0, transformers 5.18.0). Not tested: Linux, other
  GPUs, multi-GPU DDP + kernels, bigger models, a full-length run, `--fuse` in a full training run (only a tiny
  GPT-2 and the activation micro-benchmarks).
* RoPE patching replaces the model file's module-level `apply_rotary_pos_emb`, which affects every model of that
  family in the process until `restore_kernels(model)`.
* `rmsnorm` steps aside under QLoRA (fp32 norm weights, bf16 activations); `rmsnorm` backward always computes the
  weight gradient even though norm weights are frozen under LoRA (small waste).
* Fusion covers leaf modules only (fx cannot trace HF decoder layers), the op list in `docs/kernels.md`, and
  same-shape inputs; it does not fuse into attention or matmuls.
* Not in this plan at all: attention and quantised-matmul kernels (phase 3), the UI workbench, a kernels page in
  the web UI. Renting a Linux/large GPU (Vast.ai has no standard free credit; needs the user's account, API key and
  a few dollars) is the next validation step; nothing was rented.
* Local `main` is far ahead of origin from earlier work and this branch adds ~35 commits; nothing is pushed or
  published. Ask before pushing, merging or publishing to PyPI.
* I once committed `tools/kagvenv/` (a virtualenv) by running `git add -A tools`; it was undone before anything
  else happened (reset --soft, recommit). The untracked `tools/kagvenv`, `kk`, `nbpatch`, `video`, `dist/` are the
  user's, were there before this work, and must never be added with `git add -A`.

## If you resume
1. Run the three checks above. 2. Read `docs/kernels.md` (what is promised). 3. Next valuable step is validation on
Linux + a bigger GPU (`pip install tinyforge`, `tinyforge kernels doctor`, `kernels verify`, `kernels bench`, then
`--fuse`), then flip the default. 4. Phase 3 (attention, quantised matmul) only after that.
