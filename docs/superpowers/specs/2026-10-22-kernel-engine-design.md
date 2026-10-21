# Kernel engine: verified, shape-specialised Triton kernels for real training and inference

Status: draft for review (2026-10-22). Phases 1 and 2 are specified here; phase 3 is listed only as a direction.

## 1. Intent

tinyforge should create GPU kernels on demand, specialised to the user's model, shapes, dtype and GPU, and use them
in real fine-tuning (`tinyforge ft`) and real inference (`tinyforge serve`). A kernel is only used if it is proven
correct (forward and gradients) against the PyTorch reference, and only kept if it is measurably better. Anything
that fails falls back to stock PyTorch, so enabling kernels can never break a run.

Stated by the user: "it should be able to make kernels, dynamic kernels that can be used for actual training or
inferencing." Chosen approach: a verified kernel library with auto-fusion (library first, then a graph-level
fusion generator).

## 2. Current state (verified 2026-10-22)

- `src/tinyforge/kernels/rmsnorm.py` is the only kernel: Triton RMSNorm forward, no backward.
- It is called only from the toy GPT in `model.py`. The HuggingFace fine-tune path (`finetune.py`) and the
  `serve` path never use it.
- `memory.py` already has a chunked cross-entropy in plain PyTorch (the memory-ladder work); a fused kernel
  would replace its inner loop, not its API.
- Environment: RTX 3050 Ti (sm_86, 4 GB), torch 2.6.0+cu124, triton 3.8.0, Windows, no `nvcc` or MSVC.
  Triton is the only usable kernel language here. Whether Triton compiles and launches on this GPU is NOT yet
  proven by this session; the first task of phase 1 is a smoke test (section 7).

## 3. Non-goals

- Raw CUDA C++ kernels (no compiler on the dev box).
- Replacing `torch.compile`/Inductor. We do not wrap it; it needs a host C++ compiler on Windows and offers no
  per-kernel verification.
- Attention and quantised-matmul kernels (phase 3).
- Any UI work. A kernels page in the UI is a later cycle.

## 4. Architecture

Five units, each with one job and a narrow interface.

### 4.1 Registry (`kernels/registry.py`)
A `KernelSpec` describes one kernel: `name`, `reference` (pure PyTorch callable), `triton_fn` (forward+backward
packaged as a `torch.autograd.Function`), supported dtypes, a shape-validity predicate, and the tunable
parameters (e.g. `BLOCK`, `num_warps`). `register(spec)` and `get(name)` are the whole interface.
Depends on nothing but torch.

### 4.2 Verifier (`kernels/verify.py`)
`verify(spec, shapes, dtypes) -> VerifyResult`. Runs reference and kernel on random inputs, including edge shapes
(non-power-of-two widths, tiny and large rows, fp16/bf16/fp32), compares outputs AND input/weight gradients with
dtype-appropriate tolerances, and records max error. A kernel that fails any case is `rejected` for that
shape/dtype and never used there.

### 4.3 Autotuner and cache (`kernels/tune.py`)
For each (kernel, concrete shape, dtype, GPU name, triton version) tuple, search the spec's tunable space,
time candidates with CUDA events (warmup + median), and persist the winner to a JSON cache in the tinyforge
cache dir. Cache key includes GPU name and triton version so a new GPU or upgrade retunes. A kernel is
`adopted` only if its tuned time beats the reference by a configurable margin (default 5%) OR it saves memory
(for kernels whose point is memory, e.g. fused cross-entropy); otherwise the reference stays.

### 4.4 Patcher (`kernels/patch.py`)
`apply_kernels(model, mode="auto", report=...) -> PatchReport`. Walks the loaded HF model, matches modules
by class/signature (Llama/Qwen/SmolLM-style `RMSNorm`, rotary embedding, gated `MLP`, LM-head loss), and replaces
the forward with the adopted kernel. Per-module try/except: any failure during verification, tuning or the
first live call restores the original forward and records the reason. Returns what was patched, what was
skipped and why, with measured speedup per kernel. Patching works under PEFT/LoRA (the frozen base modules
are patched, adapters untouched) and in `eval()` for inference.

### 4.5 Fusion engine, phase 2 (`kernels/fuse.py`)
Uses `torch.fx` to trace a module, finds maximal chains of fusable elementwise and row-reduction ops that no
library kernel covers, and generates a Triton kernel from a small op table (add, mul, silu, gelu, rsqrt,
row-mean, ...). Generated kernels go through the same verifier, autotuner and patcher as library kernels, so
codegen bugs are caught by the equivalence gate and degrade to fallback. Out of scope for the generator:
data-dependent control flow, matmul, anything the op table does not list (those chains are left unfused).

## 5. Phase 1 kernel library

| Kernel | Forward | Backward | Main benefit |
|---|---|---|---|
| RMSNorm | yes (exists) | new | speed, fewer launches |
| Rotary embedding (RoPE) | new | new | speed |
| SwiGLU (silu(gate)*up) | new | new | memory + speed |
| Fused cross-entropy (loss + grad, no full logits) | new | new | large VRAM saving, key on 4 GB |
| Inference-side fused residual+RMSNorm | new | not needed | decode latency |

## 6. User-facing surface

- `tinyforge kernels doctor`: Triton smoke test, GPU/compile status, what would be enabled.
- `tinyforge kernels verify [--model M]`: runs the verifier on the model's real shapes.
- `tinyforge kernels bench [--model M]`: reference vs tuned kernel, per kernel, with tok/s and peak VRAM for a
  short real train step and a short generate.
- `--kernels {auto,off,force}` on `ft`, `serve` and the worker JobSpec (`spec` gains an optional
  `kernels: auto|off`). Default `auto` only after phase 1 passes acceptance; until then default `off`.
- Every run records a kernel report (patched/skipped/why, speedups) in its run directory and the telemetry.

## 6b. User-defined kernels (added 2026-10-22 at the user's request)

Users can write their own Triton kernel and have tinyforge verify, tune and use it through the same gate as the
built-in ones. A user kernel is a Python file that defines a `KernelSpec` (reference, kernel, tunables,
`make_inputs`, as in section 4.1) plus an optional `Target` that tells the patcher which modules it replaces:
`match(module) -> bool`, `shape_for(module) -> tuple[int, ...]` (what to tune on) and
`make_forward(module, params) -> Callable` (the replacement forward).

- `tinyforge kernels new NAME` writes a runnable template (a working example with reference, Triton kernel,
  `make_inputs` and a target) into `./tinyforge_kernels/NAME.py`.
- `tinyforge kernels verify --file F` and `kernels bench --file F` run the section 4.2/4.3 gate on the file's
  spec and exit non-zero if it fails verification.
- Loading is explicit only: files are read from `--kernel-path` / `TINYFORGE_KERNEL_PATH` / `./tinyforge_kernels/`
  when `--kernels` is not `off`. A JobSpec or any network input never causes code to be loaded. User kernels are
  the user's own code running with their privileges; the docs say so.
- A user kernel that fails to import, fails verification, is slower (in `auto`), or raises at runtime is skipped
  with its reason in the kernel report and the run continues on PyTorch.
- Verification only covers the shapes and dtypes the kernel's `make_inputs` is run with, so the report says
  "verified on: ..." and never claims general correctness.

## 7. Error handling

- Triton or CUDA missing/unusable: `kernels doctor` says why; runs proceed with PyTorch (existing HW006 style
  warning, not an error).
- Kernel compile failure, verify failure, or runtime exception on first live call: restore original forward,
  log a warning with the reason, continue. A run never fails because of a kernel.
- Numerical drift during training (loss differs from a reference check on step 1 by more than tolerance): the
  engine disables kernels for that run and restarts the step on the reference path.
- Out of memory during tuning: skip that candidate; tuning uses small batches and a hard memory cap.

## 8. Testing and acceptance

Unit (CPU-safe where possible, GPU tests marked and skipped without CUDA): registry round trips, verifier
catches a deliberately wrong kernel, autotune cache hit/miss/invalidate, patcher restores the original on
injected failure, fx fusion produces the same output as eager on a sample chain.

Phase 1 acceptance, on this laptop, recorded under `benchmarks/`:
1. Smoke test: existing RMSNorm kernel compiles and matches reference on the 3050 Ti.
2. Every phase-1 kernel passes forward + gradient equivalence in fp16/bf16/fp32.
3. Fine-tune SmolLM2-360M (and Qwen 1.5B 4-bit if it fits) for N steps with `kernels off` vs `auto`: loss
   curves match within tolerance; report tok/s and peak VRAM for both, whatever the result. If there is no
   speedup we say so; memory savings are the expected win.
4. `serve` generation output is token-identical (greedy) or within tolerance with and without kernels.

Phase 2 acceptance: on a model block with an unknown elementwise chain, the fusion engine generates a kernel
that passes the verifier and is adopted or rejected correctly by the benchmark gate.

## 9. Risks and open questions

- Triton on Windows (triton 3.8.0 here) may not compile some kernels, or may be slower than expected on
  sm_86 with 4 GB. Mitigated by the smoke test first and by the fallback design.
- Speedups on a small laptop GPU may be modest; claims in docs must come from the recorded benchmarks only.
- HF module class names vary across transformers versions; the patcher matches on behaviour/signature with an
  allow-list and skips unknown classes rather than guessing.
- Linux/datacenter GPU validation is deferred (per OBJECTIVES.md direction R); cache key and tests are written
  so it works there, but it is untested on Linux.
