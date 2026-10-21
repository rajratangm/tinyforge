# Kernel Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verified, shape-autotuned Triton kernels (library in phase 1, graph-level fusion generator in phase 2) that are patched into real HF fine-tuning and inference, with automatic fallback to PyTorch.

**Architecture:** A registry of `KernelSpec`s (reference op + Triton autograd Function + tunables) feeds a verifier (forward and gradient equivalence), an autotuner with a per-GPU JSON cache, and a patcher that swaps adopted kernels into a loaded HF model. Any failure restores the original forward. Phase 2 adds an `torch.fx` matcher and a small Triton code generator whose output goes through the same verify/tune/patch pipeline.

**Tech Stack:** Python 3.12, torch 2.6.0+cu124, triton 3.8.0 (Windows), transformers 5.18.0, peft 0.21.2, typer, pytest. Run tests with `.venv\Scripts\python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-10-22-kernel-engine-design.md`

## Global Constraints

- Triton is the only kernel language (no `nvcc`/MSVC on the dev box); raw CUDA C++ is out of scope.
- A kernel is used only if forward AND gradient equivalence pass; it is adopted only if tuned time beats the reference by the margin (default 5%) OR it saves memory (fused cross-entropy).
- A run must never fail because of a kernel: compile, verify and first-live-call failures restore the original forward and log the reason.
- Autotune cache key = (kernel, concrete shape, dtype, GPU name, triton version).
- `--kernels` values are `auto|off|force`; default stays `off` until Task 12 passes, then becomes `auto`.
- GPU tests are marked `@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")`, matching `tests/test_core.py`.
- Doc claims about speed or VRAM come only from files recorded under `benchmarks/`.
- No pushing; commits are local on a branch `feat/kernel-engine`.
- Existing `rmsnorm_triton(x, weight, eps)` in `kernels/rmsnorm.py` and its test in `tests/test_core.py` must keep working (used by `model.py`).

## Review Focus

1. **Non-power-of-two widths** (SmolLM2-360M hidden 960, Qwen2.5-1.5B 1536, intermediate 8960): RMSNorm/SwiGLU/CE must match the reference, not just 2^k sizes.
2. **GQA rotary**: q and k have different head counts (e.g. 14 vs 2); RoPE must rotate both correctly, and at decode (seq len 1) too.
3. **All labels ignored (-100) in a CE chunk**: loss contribution 0 and zero gradient, never NaN.
4. **LoRA-wrapped and bitsandbytes 4-bit modules**: patcher must not break them; SwiGLU must call the wrapped `gate_proj`/`up_proj` and fuse only the activation-multiply.
5. **Gradient checkpointing recompute**: kernels must be stateless/deterministic so a recompute in backward gives identical gradients.

---

### Task 1: Triton smoke test and `kernels doctor`

**Files:**
- Create: `src/tinyforge/kernels/env.py`, `tests/test_kernels_env.py`
- Modify: `src/tinyforge/cli.py` (add `kernels_app` Typer group `kernels`, command `doctor`)

**Interfaces:**
- Produces: `triton_status() -> KernelEnv` where `KernelEnv` is a dataclass `(ok: bool, reason: str, gpu: str, compute_capability: str, triton_version: str)`; `ok` is True only if a trivial `@triton.jit` add kernel compiles, launches and matches `a + b` on the current CUDA device. Never raises.

- [ ] **Step 1: Write failing tests** `test_status_never_raises_without_cuda` (monkeypatch `torch.cuda.is_available` to False; assert `ok is False` and `reason` non-empty) and a CUDA-gated `test_smoke_kernel_runs` asserting `triton_status().ok is True`.
- [ ] **Step 2: Run** `.venv\Scripts\python -m pytest tests/test_kernels_env.py -v`; expect FAIL (module missing).
- [ ] **Step 3: Implement** `triton_status` in `env.py`; catch every exception into `reason`. Add `tinyforge kernels doctor` printing the fields plus which phase-1 kernels would be enabled (empty list until Task 8).
- [ ] **Step 4: Run tests** (expect PASS) and run `.venv\Scripts\tinyforge kernels doctor` on the laptop.
- [ ] **Step 5: Record the real outcome** in `benchmarks/kernels-smoke.json` (ok/reason/gpu/triton version). **If `ok` is False, STOP and report to the user: the plan's premise (Triton on this GPU) needs rethinking.**
- [ ] **Step 6: Commit** `feat(kernels): triton smoke test and kernels doctor`.

### Task 2: Registry

**Files:**
- Create: `src/tinyforge/kernels/registry.py`, `tests/test_kernels_registry.py`

**Interfaces:**
- Produces: `@dataclass KernelSpec(name: str, reference: Callable, kernel: Callable, dtypes: tuple[torch.dtype, ...], supports: Callable[[tuple[int, ...]], bool], tunables: dict[str, list[int]], make_inputs: Callable[[tuple[int, ...], torch.dtype, str], tuple[torch.Tensor, ...]], memory_kernel: bool = False)`; `register(spec) -> None` (duplicate name raises `ValueError`); `get(name) -> KernelSpec` (`KeyError` if unknown); `all_specs() -> list[KernelSpec]`. `kernel` and `reference` take the same positional args from `make_inputs` and return tensors (or a tuple of tensors); `kernel` additionally accepts `**tune` keyword args drawn from `tunables`.

- [ ] **Step 1: Write failing tests** `test_register_and_get`, `test_duplicate_register_raises`, `test_unknown_get_raises`.
- [ ] **Step 2: Run** the test file; expect FAIL.
- [ ] **Step 3: Implement** the dataclass and module-level dict registry.
- [ ] **Step 4: Run** the tests; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): kernel registry`.

### Task 3: Verifier

**Files:**
- Create: `src/tinyforge/kernels/verify.py`, `tests/test_kernels_verify.py`

**Interfaces:**
- Consumes: `KernelSpec` (Task 2).
- Produces: `@dataclass VerifyResult(ok: bool, max_err: float, grad_err: float, reason: str)`; `verify(spec, shape: tuple[int, ...], dtype: torch.dtype, device: str = "cuda", **tune) -> VerifyResult`. Inputs from `spec.make_inputs`; float inputs get `requires_grad_()`; compares outputs and every input gradient (via `sum(out * fixed_random_cotangent).backward()`). Tolerances: fp32 1e-5 abs, bf16 2e-2, fp16 5e-3, all scaled by `max(1, ref.abs().max())`. Exceptions inside the kernel become `ok=False` with the exception text in `reason`.

- [ ] **Step 1: Write failing tests** using a CPU-runnable toy spec (`reference=lambda x: x*2`, `kernel=lambda x, **t: x*2`): `test_correct_kernel_passes`; `test_wrong_kernel_fails` (kernel returns `x*2.1`, assert `ok is False`); `test_wrong_gradient_fails` (kernel is a custom `autograd.Function` with a deliberately wrong backward, assert `ok is False` and `grad_err > 0`); `test_kernel_exception_is_captured`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** `verify` (device parameter lets the tests run on CPU).
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): equivalence verifier (outputs and gradients)`.

### Task 4: Autotuner and cache

**Files:**
- Create: `src/tinyforge/kernels/tune.py`, `tests/test_kernels_tune.py`

**Interfaces:**
- Consumes: `KernelSpec`, `verify`.
- Produces: `@dataclass TuneResult(params: dict[str, int], kernel_ms: float, ref_ms: float, adopted: bool, reason: str)`; `tune(spec, shape, dtype, device="cuda", margin: float = 0.05, cache_dir: Path | None = None, time_fn: Callable | None = None) -> TuneResult`. Candidates = cartesian product of `spec.tunables`; each must pass `verify` first; timing via `time_fn(callable) -> ms` (default: CUDA events, 3 warmup, median of 10; injectable so tests run on CPU). Cache file `<cache_dir>/kernel_tune.json` (default `~/.cache/tinyforge`, overridable with env `TINYFORGE_CACHE`), key string `f"{name}|{shape}|{dtype}|{gpu}|{triton_version}"`. Adopted iff verified AND (`kernel_ms <= ref_ms*(1-margin)` OR `spec.memory_kernel`). Rejections are cached too.

- [ ] **Step 1: Write failing tests** with fake `time_fn`: `test_picks_fastest_candidate`, `test_not_adopted_when_slower`, `test_memory_kernel_adopted_even_if_slower`, `test_cache_hit_skips_timing` (assert `time_fn` call count unchanged on second call), `test_cache_key_changes_with_gpu_name` (monkeypatch gpu name), `test_failing_verify_candidates_skipped`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** `tune` and the cache read/write (atomic write via temp file + `os.replace`).
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): autotuner with per-gpu cache`.

### Task 5: RMSNorm forward + backward

**Files:**
- Modify: `src/tinyforge/kernels/rmsnorm.py` (keep `rmsnorm_triton`; add backward and a spec)
- Test: `tests/test_kernels_rmsnorm.py`

**Interfaces:**
- Produces: `rmsnorm_fn(x: Tensor, weight: Tensor, eps: float = 1e-6, BLOCK: int | None = None, num_warps: int = 4) -> Tensor` (a `torch.autograd.Function` wrapper; grads for x and weight). Semantics must equal HF: normalise in fp32, cast to input dtype, THEN multiply by weight. Registers `KernelSpec("rmsnorm", ...)` with `tunables={"num_warps": [2, 4, 8]}`, dtypes fp16/bf16/fp32, shapes `(rows, hidden)`. Reference is a copy of the HF formula (not imported from transformers).

- [ ] **Step 1: Write failing tests** (CUDA-gated, via `verify`): `test_rmsnorm_equivalence` parametrised over hidden `[96, 960, 1536, 2048]` x dtypes, asserting `verify(...).ok`; `test_weight_grad_matches` ; `test_checkpoint_recompute_identical` (run under `torch.utils.checkpoint`, compare grads to non-checkpointed).
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** forward (existing kernel adjusted to HF cast order) and a backward kernel (row-wise dx; dweight via per-program partial sums reduced in torch). Register the spec at import.
- [ ] **Step 4: Run** the new tests and `tests/test_core.py::test_triton_rmsnorm_matches_torch`; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): rmsnorm forward+backward matching HF semantics`.

### Task 6: SwiGLU (activation-multiply) kernel

**Files:**
- Create: `src/tinyforge/kernels/swiglu.py`, `tests/test_kernels_swiglu.py`

**Interfaces:**
- Produces: `swiglu_fn(gate: Tensor, up: Tensor, BLOCK: int = 1024, num_warps: int = 4) -> Tensor` computing `silu(gate) * up` elementwise with an autograd backward (dgate, dup); spec `"swiglu"`, shape `(rows, intermediate)`, `memory_kernel=True` (no stored silu intermediate).

- [ ] **Step 1: Write failing tests** CUDA-gated: equivalence over intermediates `[256, 8960, 8193]`, three dtypes, via `verify`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** forward and backward Triton kernels (flatten to 1-D, mask the tail) and register.
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): fused silu*up with backward`.

### Task 7: RoPE kernel

**Files:**
- Create: `src/tinyforge/kernels/rope.py`, `tests/test_kernels_rope.py`

**Interfaces:**
- Produces: `rope_fn(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1, num_warps: int = 4) -> tuple[Tensor, Tensor]` with the same semantics as HF `apply_rotary_pos_emb` (half-rotation, `cos`/`sin` shape `[batch, seq, head_dim]`, q `[batch, heads_q, seq, head_dim]`, k `[batch, heads_k, seq, head_dim]`), autograd backward for q and k; spec `"rope"`.

- [ ] **Step 1: Write failing tests** CUDA-gated: equivalence with GQA `(heads_q, heads_k) = (14, 2)` and `(8, 8)`, `seq in {1, 33, 128}`, head_dim `[64, 128]`; including seq=1 decode.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** the in-place-free rotation kernel (one program per (batch, head, position)) and backward (rotation by negated sin).
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): rotary embedding forward+backward with GQA`.

### Task 8: Fused cross-entropy

**Files:**
- Create: `src/tinyforge/kernels/ce.py`, `tests/test_kernels_ce.py`
- Modify: `src/tinyforge/memory.py` (`chunked_ce_loss` calls the fused path when enabled)

**Interfaces:**
- Produces: `fused_ce_chunk_sum(hidden: Tensor, weight: Tensor, bias: Tensor | None, labels: Tensor, num_warps: int = 8) -> Tensor` returning the summed (not mean) loss over non-`-100` labels for one chunk, as an autograd Function: forward computes `logits = hidden @ weight.T` then a Triton kernel produces per-row loss and writes `dlogits` in place of the logits buffer (no fp32 copy of the logits); backward uses the saved `dlogits`. Spec `"fused_ce"`, `memory_kernel=True`. `chunked_ce_loss` gains parameter `fused: bool = False` that swaps `_chunk_ce_sum` for this.

- [ ] **Step 1: Write failing tests** CUDA-gated: `test_ce_equivalence` vs `_chunk_ce_sum` with vocab `[1000, 49152, 151936]`; `test_all_ignored_is_zero_and_no_nan` (labels all `-100`: loss `== 0`, grads finite and zero); `test_peak_memory_below_reference` (assert `torch.cuda.max_memory_allocated` lower than the reference at vocab 151936, chunk 512).
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** kernel and Function; thread `fused` through `chunked_ce_loss` and `causal_lm_loss`.
- [ ] **Step 4: Run** new tests and existing memory tests (`tests/test_memory.py` or equivalent); expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): fused cross-entropy chunk with in-place logit gradient`.

### Task 9: Fused residual + RMSNorm (inference)

**Files:**
- Create: `src/tinyforge/kernels/add_rmsnorm.py`, `tests/test_kernels_add_rmsnorm.py`

**Interfaces:**
- Produces: `add_rmsnorm(x: Tensor, residual: Tensor, weight: Tensor, eps: float, num_warps: int = 4) -> tuple[Tensor, Tensor]` returning `(normed, new_residual)` where `new_residual = x + residual` (forward only; spec `"add_rmsnorm"` with `kernel` documented as inference-only, so verify with `torch.no_grad()` via a spec flag `needs_grad: bool = True` added to `KernelSpec` in this task, default True; the verifier skips gradient comparison when False).

- [ ] **Step 1: Write failing tests**: equivalence (hidden `[960, 1536, 2048]`, three dtypes) and `test_verifier_skips_grad_when_not_needed` (CPU toy spec).
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** kernel, add `needs_grad` to `KernelSpec` and honor it in `verify`.
- [ ] **Step 4: Run** this file plus Tasks 2-3 tests; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): fused add+rmsnorm for inference`.

### Task 10: Patcher

**Files:**
- Create: `src/tinyforge/kernels/patch.py`, `tests/test_kernels_patch.py`

**Interfaces:**
- Consumes: registry, `verify`, `tune`, `triton_status`.
- Produces: `@dataclass KernelReport(patched: list[str], skipped: dict[str, str], speedups: dict[str, float])`; `apply_kernels(model: nn.Module, mode: str = "auto", inference: bool = False) -> KernelReport`; `mode="off"` returns an empty report without touching the model. Matching is allow-list by class name: RMSNorm = classes named `*RMSNorm` with `weight` and `variance_epsilon`; MLP = classes with `gate_proj`, `up_proj`, `down_proj`, `act_fn` where `act_fn` is SiLU; RoPE by wrapping the module-level `apply_rotary_pos_emb` in the model's defining module; fused CE is applied by Task 11, not here; with `inference=True`, decoder layers of the allow-listed Llama/Qwen2/Mistral classes get a forward that uses `add_rmsnorm` only when `torch.is_grad_enabled()` is False. For each match, `verify` + `tune` on the actual `(rows, width)` once; not adopted means skipped with the tune reason. Patched forwards wrap the kernel call in try/except: on exception, restore the original forward permanently and record the reason in the report (module-level `report` object kept on the model as `model._tinyforge_kernel_report`). `force` skips the speed margin but never skips verify. SwiGLU patch calls the module's own (possibly LoRA-wrapped) `gate_proj`, `up_proj`, `down_proj` and replaces only `act_fn(gate)*up` with `swiglu_fn`.

- [ ] **Step 1: Write failing tests** using a tiny randomly-initialised `Qwen2Config` model (hidden 64, 2 layers, built in-test, no download): `test_off_changes_nothing`; `test_no_triton_returns_all_skipped_no_exception` (monkeypatch `triton_status` to not ok); `test_patched_model_outputs_match` (CUDA-gated, logits within tolerance and gradients equal before/after); `test_failure_restores_original` (inject a kernel that raises on call; assert second forward still works and report records the reason); `test_lora_wrapped_mlp_still_trains` (wrap with `get_peft_model`, patch, run one backward, assert adapter grads non-zero); `test_gqa_rope_patched_matches` (CUDA-gated).
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** `apply_kernels`, matchers and the restore-on-failure wrapper.
- [ ] **Step 4: Run**; expect PASS. Extend `kernels doctor` to list which matchers apply to a `--model` config.
- [ ] **Step 5: Commit** `feat(kernels): patcher with verify, tune and automatic fallback`.

### Task 11: Wire into fine-tuning and serving, with drift guard

**Files:**
- Modify: `src/tinyforge/finetune.py` (`FTConfig.kernels: str = "off"`; apply after `get_peft_model` at `train()`; pass `fused=True` to the loss when CE kernel is adopted), `src/tinyforge/cli.py` (`--kernels` on `ft train`, `ft pipeline`, `serve`), `src/tinyforge/server.py`/`engines.py` (serve path calls `apply_kernels(model, mode, inference=True)`), `src/tinyforge/jobspec.v1alpha1.schema.json` and `spec/` copy plus the worker mapping (optional field `kernels: auto|off|force`).
- Test: `tests/test_kernels_integration.py`

**Interfaces:**
- Consumes: `apply_kernels`, `KernelReport`.
- Produces: `FTConfig.kernels`; run directory file `kernel_report.json` (the `KernelReport` as dict); a training `event("kernels", **report)`; drift guard `kernel_drift_ok(model, batch, ce_chunk) -> bool` in `kernels/patch.py` comparing step-1 loss with kernels vs reference (relative tolerance 2%), and on failure the training loop restores originals (`restore_kernels(model)`) and logs warning code `FT0xx` (next free code in `warnings`) and re-runs the step.

- [ ] **Step 1: Write failing tests**: `test_ftconfig_kernels_default_off`; `test_jobspec_accepts_kernels_field` (and still rejects bad values) using the existing schema tests' pattern; `test_train_writes_kernel_report` (CUDA-gated, 5-step run on a tiny local model if the existing finetune tests have a fixture; otherwise mark skip with reason); `test_drift_guard_disables_on_mismatch` (inject a kernel that returns wrong values after verification).
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** wiring, schema change (also run the Go `forgectl validate` tests with `GOROOT`/`GOPATH` from `tools/` since the schema is shared), and the drift guard.
- [ ] **Step 4: Run** `.venv\Scripts\python -m pytest -q` (whole suite) and `ruff check`; expect all green.
- [ ] **Step 5: Commit** `feat(kernels): --kernels flag across ft, serve and jobspec, with drift guard`.

### Task 12: `kernels verify|bench` and phase 1 acceptance

**Files:**
- Create: `src/tinyforge/kernels/bench.py`, `benchmarks/kernels-phase1.json` (generated), `docs/kernels.md`
- Modify: `src/tinyforge/cli.py` (`kernels verify`, `kernels bench`), `README.md` (short pointer under "Verified results"), `CHANGELOG.md`
- Test: `tests/test_kernels_bench.py`

**Interfaces:**
- Produces: `bench_model(model_name: str, steps: int = 30, quant: str = "none") -> dict` returning `{"off": {"tok_s", "peak_vram_gb", "final_loss"}, "auto": {...}, "kernel_report": {...}}`; `kernels verify --model M` runs `verify` over the model's real shapes and prints a table.

- [ ] **Step 1: Write failing tests**: `test_bench_result_schema` with a stubbed runner (keys present, `final_loss` equal within tolerance flag computed), and `test_verify_cli_exit_code_nonzero_on_failure` (typer `CliRunner` with a registered always-wrong spec).
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** the commands. Then run the real acceptance on the laptop: SmolLM2-360M-Instruct, 30 steps, `off` vs `auto`; then Qwen2.5-1.5B 4-bit if memory allows (stop other GPU users first); then a greedy `ft generate` comparison with and without kernels.
- [ ] **Step 4: Check acceptance** (spec section 8): kernels pass equivalence; loss curves match within 2%; generations identical or within tolerance. Write results to `benchmarks/kernels-phase1.json` exactly as measured, including a no-speedup result if that is what happened. Document honestly in `docs/kernels.md`.
- [ ] **Step 5: Only if acceptance passed**, change the `--kernels` default to `auto` in `FTConfig`/CLI and update the Global Constraint note; otherwise leave `off` and record why.
- [ ] **Step 6: Commit** `feat(kernels): verify/bench commands and phase 1 acceptance results`.

### Task 13 (phase 2): Op table and Triton code generator

**Files:**
- Create: `src/tinyforge/kernels/codegen.py`, `tests/test_kernels_codegen.py`

**Interfaces:**
- Produces: `OP_TABLE: dict[str, OpDef]` covering elementwise `add, sub, mul, div, neg, silu, gelu, sigmoid, tanh, exp, rsqrt, relu` and row reduction `mean`/`sum` over the last dim; `generate_elementwise(chain: list[ChainNode]) -> GeneratedKernel` with `.source: str`, `.kernel: Callable` (compiled from `source`, with a fixed BLOCK tunable) and `.reference: Callable` (the original eager ops). `ChainNode(op: str, inputs: list[int | str], attrs: dict)`. Generated sources are written to the cache dir (`generated/<hash>.py`) so they are inspectable.

- [ ] **Step 1: Write failing tests** CUDA-gated: `test_generated_silu_mul_matches_eager`, `test_generated_chain_with_broadcast_scalar`, `test_unknown_op_raises_unsupported`, `test_same_chain_same_hash_cache_reused`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** codegen (string templating of a `@triton.jit` function, `exec` into a module file under the cache dir, autograd backward generated by symbolic derivative per op from the table; ops without a derivative mark the chain inference-only).
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): triton code generator for elementwise chains`.

### Task 14 (phase 2): fx chain matcher

**Files:**
- Create: `src/tinyforge/kernels/fuse.py`, `tests/test_kernels_fuse.py`

**Interfaces:**
- Consumes: `OP_TABLE`, `generate_elementwise`.
- Produces: `find_fusable_chains(module: nn.Module, example_args: tuple) -> list[ChainNode-list]` using `torch.fx.symbolic_trace` (modules that fail to trace are reported as skipped with the reason, never raised); chains are maximal connected runs of table ops with a single consumer between nodes and length >= 2 that no registry kernel already covers (name-pattern check against `swiglu`, `rmsnorm`).

- [ ] **Step 1: Write failing tests** with a small custom `nn.Module` (e.g. `x*torch.sigmoid(x)+bias`): `test_finds_chain`, `test_untraceable_module_reported_not_raised` (data-dependent `if`), `test_registry_covered_pattern_excluded`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** the matcher.
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(kernels): fx-based fusable chain finder`.

### Task 15 (phase 2): Fusion in the patcher and phase 2 acceptance

**Files:**
- Modify: `src/tinyforge/kernels/patch.py` (`apply_kernels(..., fuse: bool = False)`), `src/tinyforge/cli.py` (`kernels fuse --model M`), `docs/kernels.md`
- Test: `tests/test_kernels_fuse_integration.py`

**Interfaces:**
- Consumes: `find_fusable_chains`, `generate_elementwise`, `verify`, `tune`.
- Produces: with `fuse=True`, each found chain becomes a generated `KernelSpec` (name `fused_<hash>`), goes through verify+tune, and is patched in via the same restore-on-failure wrapper; report entries appear in `KernelReport.patched/skipped` under the generated name.

- [ ] **Step 1: Write failing tests**: `test_unknown_chain_is_generated_and_adopted_or_rejected` (CUDA-gated: module with a custom activation chain; assert outputs equal before/after and the report names a `fused_*` entry as patched or skipped with a reason); `test_bad_generated_kernel_falls_back` (monkeypatch codegen to emit a wrong kernel; assert verify rejects it and the model output is unchanged).
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** integration and `kernels fuse`.
- [ ] **Step 4: Run** the full suite and ruff; expect green. Record the phase 2 result under `benchmarks/kernels-phase2.json` and document it in `docs/kernels.md`, noting what the op table does not cover.
- [ ] **Step 5: Commit** `feat(kernels): auto-fusion of unknown elementwise chains`.

---

## Self-review notes

- Spec coverage: registry (T2), verifier (T3), autotune+cache (T4), patcher+fallback (T10), fx fusion (T13-15), phase 1 kernels (T5-9), CLI surface (T1, T10, T12, T15), JobSpec/`--kernels` (T11), error handling incl. drift guard (T10-11), acceptance (T1, T12, T15). Phase 3 (attention, quantised matmul) is intentionally absent.
- Known unknowns: Task 1 may fail on this GPU; Task 11 depends on the existing finetune test fixtures and on the shared schema (Go side); Task 8 ties into memory.py's existing API.
- Proportion: plan is about 2x the spec, mostly signatures, tests and checks; kernel bodies are left to the implementer.

### Task 16 (added at the user's request; run after Task 12): User-defined kernels

Spec section 6b.

**Files:**
- Create: `src/tinyforge/kernels/user.py`, `examples/kernels/gelu_mul.py`, `tests/test_kernels_user.py`
- Modify: `src/tinyforge/kernels/registry.py` (`Target` dataclass), `src/tinyforge/kernels/patch.py` (apply user targets), `src/tinyforge/cli.py` (`kernels new`, `--file` on `kernels verify|bench`, `--kernel-path`), `docs/kernels.md`

**Interfaces:**
- Consumes: `KernelSpec`, `verify`, `tune`, `_wrap_forward`, `apply_kernels` (Tasks 2-4, 10).
- Produces: `@dataclass Target(match: Callable[[nn.Module], bool], shape_for: Callable[[nn.Module], tuple[int, ...]], make_forward: Callable[[nn.Module, dict], Callable])`; `KernelSpec.target: Target | None = None`; `load_user_kernels(paths: list[Path]) -> tuple[list[KernelSpec], dict[str, str]]` (specs found, and `{file: error}` for files that failed to import; never raises); `kernel_search_path(cli_value: Path | None) -> list[Path]` (CLI value, then `TINYFORGE_KERNEL_PATH` split on `os.pathsep`, then `./tinyforge_kernels` if it exists); `scaffold(name: str, out_dir: Path) -> Path`; `apply_kernels(..., user_paths: list[Path] | None = None)` where user targets go through the same `tune` + `_wrap_forward` path and appear in the report under `user:<name>`.

- [ ] **Step 1: Write failing tests** in `tests/test_kernels_user.py`: `test_load_registers_spec_from_file` (tmp file defining `SPEC` with CPU toy kernel; assert returned and not leaked into the global registry on repeat loads); `test_broken_file_is_reported_not_raised` (syntax error file gives `{file: msg}`, other files still load); `test_scaffold_writes_a_loadable_file_that_passes_verify` (scaffold, load, `verify(...).ok`, CPU device); `test_cli_verify_file_exits_nonzero_for_wrong_kernel` (file with `x*2.1`, `CliRunner` exit code 1 and output contains `verified on`); `test_user_kernel_is_not_loaded_unless_a_path_is_given` (monkeypatch cwd with a `tinyforge_kernels/` dir holding a file that writes a marker; `apply_kernels(model, "off")` and a JobSpec-driven `to_ft_config` leave the marker unwritten); CUDA-gated `test_user_target_is_patched_and_matches` (toy `nn.Module` named `GeluMul` in a tiny model, user kernel from `examples/kernels/gelu_mul.py`, assert `user:gelu_mul` in `report.patched` and outputs match) and `test_user_kernel_runtime_failure_falls_back` (kernel raises on call; model output still equals the original and the reason is in `report.skipped`).
- [ ] **Step 2: Run** `.venv\Scripts\python -m pytest tests/test_kernels_user.py -q --no-cov`; expect FAIL (module missing).
- [ ] **Step 3: Implement** `Target`, `load_user_kernels` (import each `*.py` with `importlib.util.spec_from_file_location` under a unique module name, read module-level `SPEC`, catch every exception per file), `kernel_search_path`, `scaffold` (template is `examples/kernels/gelu_mul.py` copied with the name replaced), and the patcher branch for user targets; wire the CLI.
- [ ] **Step 4: Run** the new tests plus `tests/test_kernels_patch.py`; expect PASS.
- [ ] **Step 5: Document** in `docs/kernels.md`: the file format, the template, the safety note (your own code, loaded only from paths you give), and what "verified on" does and does not mean.
- [ ] **Step 6: Commit** `feat(kernels): user-defined kernels with the same verify/tune/fallback gate`.
