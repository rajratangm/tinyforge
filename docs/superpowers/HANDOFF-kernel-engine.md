# Kernel engine: handoff (written 2026-10-22, session budget ending)

Read this first when resuming. Branch: `feat/kernel-engine` (local only, **nothing pushed**, not merged to `main`).
Spec: `docs/superpowers/specs/2026-10-22-kernel-engine-design.md`. Plan:
`docs/superpowers/plans/2026-10-22-kernel-engine.md` (Tasks 1-16). User-facing doc: `docs/kernels.md`.
Ledger (git-ignored scratch): `.superpowers/sdd/2026-10-22-kernel-engine/progress.md`.

## Why this exists
The user said tinyforge "does nothing" for kernels and wants it to create dynamic kernels that are used in real
training and inference. Findings at the start: the only kernel was a 27-line forward-only RMSNorm used by the toy GPT,
not by the HF fine-tune/serve path; the UI is one 280-line HTML file. Chosen direction: a verified kernel library
with auto-fusion later (library first), Triton only (no nvcc/MSVC on the laptop; JAX parked: no native Windows GPU).

## State

| Task | Status |
|---|---|
| 1 Triton smoke test + `kernels doctor` | done; Triton 3.8.0 works on the RTX 3050 Ti (`benchmarks/kernels-smoke.json`) |
| 2 registry, 3 verifier, 4 autotuner+cache | done |
| 5 RMSNorm fwd+bwd, 6 SwiGLU, 7 RoPE, 8 fused cross-entropy, 9 add+RMSNorm (inference) | done |
| 10 patcher (`apply_kernels`, `restore_kernels`, fallback) | done |
| 11 `--kernels` on ft/serve/JobSpec/CRD/Go + drift guard | done |
| 12 `kernels verify` / `kernels bench` + acceptance | **partly done**, see below |
| 13-15 phase 2: op table + Triton codegen, fx chain finder, fusion in patcher | not started |
| 16 user-defined kernels (added at the user's request, after Task 12) | not started (spec section 6b) |

Full suite at Task 11: 439 passed, 1 skipped; ruff clean; `go vet`/`go test ./...` ok. Run after any change:
`.venv\Scripts\python -m pytest -q --no-cov` and `.venv\Scripts\ruff check src tests`.

## Task 12: what remains (do in this order)
1. **Benchmark 360M: DONE after `a9c22f1`** (`benchmarks/kernels-phase1-smollm2-360m.json`: +14.5% tok/s, 2.19 to 1.34 GB peak, same loss; see `docs/kernels.md` Results). Re-run only if kernels change.
2. Same for Qwen2.5-1.5B 4-bit (`--model Qwen/Qwen2.5-1.5B-Instruct --quant 4bit`, data in `data/sql` or `data/ft`
   prepared for that tokenizer) if it fits; expect `rmsnorm` to fall back under 4-bit (fp32 norm weights).
3. Greedy `ft generate` with `--kernels off` vs `auto` against a trained adapter (`runs/kernel-bench/auto`).
   Seen so far: outputs agree for the first ~25 tokens then one word differs (bf16 rounding); document, do not hide.
4. Only if 1-3 pass: change the `--kernels` default to `auto` (`FTConfig.kernels`, CLI `KernelsOpt` defaults,
   schema/CRD/Go default), update `docs/kernels.md` Results, `README.md` "Verified results", `CHANGELOG.md`.
   Otherwise leave `off` and record why.
5. Run Task 16 (plan has the exact steps), then phase 2 (Tasks 13-15) if budget allows.

## Findings during execution (all fixed with a failing test first, except where noted)
1. `apply_kernels`: the SwiGLU forward read the loop-shared variable `params` (late binding); after RoPE reassigned
   it, SwiGLU raised `KeyError: 'BLOCK'` and silently fell back. Test:
   `test_no_kernel_silently_falls_back_during_a_real_forward_and_backward`.
2. Gradient checkpointing (the planner turns it on for 4 GB): torch's non-reentrant checkpoint raises
   `_StopRecomputationError` through the patched forward; the fallback `except Exception` swallowed it and training
   crashed. Test: `test_gradient_checkpointing_neither_trips_the_fallback_nor_changes_gradients`. Lesson: the
   fallback design can hide a broken kernel, so tests must assert nothing fell back (`report.skipped == {}`).
3. HF passes cos/sin of batch 1 during training; the RoPE kernel indexed them per batch (silent wrong values).
   Fixed with a batch stride; non-broadcastable shapes now raise.
4. `tune._runner` timed the reference fwd+bwd but inference-only kernels forward-only; now honours `needs_grad`.
5. `kernels bench` could leave an older run's JSON behind after a crash; now deletes it first.
6. Not a bug: the drift-guard test needed a large sabotage (x50) because a randomly initialised tiny model has
   near-zero logits.

## Rulings made on the user's behalf
Branch in place instead of a worktree (venv/CUDA/tools live in the main checkout). `time_fn(fn, label)` signature in
the tuner. RMSNorm follows the HF cast order. Fused CE is tuned on a proxy width (<= 512) to fit 4 GB and passed to
the loss as parameters rather than patched as a module. The drift guard bypasses patches with an `enabled` flag.
`ft_data`/chat-template audit and the other earlier follow-ups in `OBJECTIVES.md` were not touched.

## Known gaps and risks
* Everything is verified only on this Windows laptop (RTX 3050 Ti, torch 2.6.0+cu124, triton 3.8.0, transformers
  5.18.0). Linux, other GPUs, multi-GPU DDP + kernels, QLoRA + kernels are untested.
* `test_core.py`'s old toy-GPT RMSNorm path (`rmsnorm_triton`) is unchanged and still used by `model.py`.
* The tune cache lives in `~/.cache/tinyforge/kernel_tune.json` (override `TINYFORGE_CACHE`); tests redirect it.
* Task 11 tests the training wiring via `_enable_kernels` and the drift guard separately; there is no automated test
  that runs a full `finetune.train` with kernels (the `kernels bench` acceptance run is that test).
* Local `main` is still ~55+ commits ahead of origin per earlier notes, nothing published; do not push or publish to
  PyPI without asking the user.
* Superseded by this work: the user's wish for a UI workbench and attention/quantised-matmul kernels (phase 3) is
  still open; see the spec's non-goals.
