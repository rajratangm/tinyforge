# Kernels: verified GPU speed-ups for training and serving

A **kernel** is a small program that runs one step of the maths on the GPU. A model does the same few steps
millions of times (normalise, activate, rotate, compute the loss), so a leaner version of those steps makes
training faster and uses less GPU memory without changing the answer.

tinyforge ships Triton kernels for the steps below and **only uses one if it proves itself first**:

1. it must give the same output **and the same gradients** as the plain PyTorch step (the *verifier*),
2. in `auto` mode it must also be faster, or save memory (the *autotuner*),
3. if anything goes wrong while running, the original PyTorch step is put back and the run continues.

Status: **opt-in, default off** (`--kernels off`). Recommended on an NVIDIA GPU: `--kernels auto`. The default
stays `off` because every result below comes from one Windows laptop (RTX 3050 Ti); it flips to `auto` once the same
runs have been repeated on Linux and on a second GPU. See "What is and is not verified".

## Use it

```bash
tinyforge kernels doctor            # can Triton compile and run on this machine?
tinyforge kernels verify            # check every kernel against PyTorch on SmolLM2-360M shapes
tinyforge kernels verify --model Qwen/Qwen2.5-1.5B-Instruct   # ... on that model's real shapes
tinyforge kernels bench             # train the same short run with kernels off and on, compare
tinyforge ft train --kernels auto   # use them while fine-tuning   (off | auto | force)
tinyforge ft generate "..." --kernels auto
tinyforge serve --kernels auto      # the fine-tuned model served by the UI/API
```

In a JobSpec: `spec.kernels: off | auto | force`.

* `auto`: use a kernel only if it matches PyTorch and is at least 5% faster, or saves memory.
* `force`: use every kernel that matches PyTorch, even if it is not faster (for testing).
* `off`: plain PyTorch.

Each training run writes `kernel_report.json` in its run directory: what was patched, what was skipped and why,
and the measured per-kernel speed-up.

## The kernels

| Kernel | Replaces | Forward | Backward | Main benefit |
|---|---|---|---|---|
| `rmsnorm` | the RMSNorm layers | yes | yes | fewer launches, speed |
| `swiglu` | `silu(gate) * up` inside the MLP (the matrix multiplies and LoRA adapters are untouched) | yes | yes | no stored intermediate, memory |
| `rope` | rotary position embedding, including grouped-query attention (q and k with different head counts) | yes | yes | speed |
| `fused_ce` | the cross-entropy loss over one chunk of positions | yes | yes | big memory saving (no fp32 copy of the logits) |
| `add_rmsnorm` | residual add + RMSNorm in decoder layers, **inference only** | yes | no | decode latency |

They match the Hugging Face implementation of Llama/Qwen2/Mistral style models (names matched on an allow-list;
unknown model classes are left alone, never guessed).

## How it works

```
kernels/registry.py   KernelSpec: PyTorch reference + Triton kernel + tunable settings + test inputs
kernels/verify.py     runs reference and kernel, compares outputs AND input/weight gradients per dtype
kernels/tune.py       tries each setting (num_warps, block size), times it with CUDA events, caches the winner
                      per (kernel, shape, dtype, GPU name, triton version) in ~/.cache/tinyforge/kernel_tune.json
kernels/patch.py      apply_kernels(model): finds matching modules, swaps in the winners, restores on failure;
                      kernels_agree(): the drift guard; decide_fused_ce(): picks the fused loss
kernels/bench.py      `kernels verify` / `kernels bench`
kernels/env.py        `kernels doctor`: compiles and launches one tiny Triton kernel
```

Safety nets, all tested:

* **Fallback.** A kernel that fails to compile, fails verification, or raises while running is removed and the
  original forward is used. The reason is in the kernel report.
* **Drift guard.** Before step 0 of a fine-tune the loss is computed with and without kernels on the first batch;
  if they differ by more than 2% all kernels are switched off for the run (diagnostic `FT011`).
* **End-to-end check for serving.** After patching for inference the logits on a small probe are compared with the
  unpatched model; a mismatch restores everything.
* **Gradient checkpointing.** PyTorch's non-reentrant checkpoint stops its recompute by raising an internal
  exception through the forward; patched forwards let it pass (this crashed training before it was fixed).

## Auto-fusion: kernels nobody wrote by hand (`--fuse`)

Some models contain chains of small operations no library kernel covers, for example GPT-2's `gelu_new`
(`0.5 * x * (1 + tanh(0.79788 * (x + 0.044715 * x**3)))`, eight separate GPU launches in plain PyTorch). With
`--fuse` tinyforge finds such chains and **generates** one Triton kernel (forward and backward) for each:

```bash
tinyforge kernels fuse --model gpt2                     # what would be generated and used for this model
tinyforge ft train ... --kernels auto --fuse            # use it while training
tinyforge serve --kernels auto --fuse
```

How: `kernels/fuse.py` traces leaf modules with `torch.fx` (the inputs come from one small probe forward),
groups ops the generator knows (`add sub mul div neg exp log sqrt rsqrt square cube sigmoid tanh silu relu
gelu gelu_tanh`, and `mean`/`sum` over the last dimension) into chains whose intermediates are used nowhere else,
and `kernels/codegen.py` writes the Triton source to `~/.cache/tinyforge/generated/<hash>.py` (readable; the hash
covers the source text, so a changed generator never reuses an old kernel). Each generated kernel then goes through
the same verifier, autotuner and patcher as every other kernel: it is checked against the original module
(outputs and gradients), tuned, used only if faster, and removed if it ever fails.

Not fused (left exactly as they are): chains containing matmuls, ops outside the list above (for example
`softplus`, so `mish` stays eager), inputs that need broadcasting (a weight vector shared across rows), modules
`torch.fx` cannot trace (data-dependent control flow), and patterns the library already covers by hand
(`silu(gate) * up`, RMSNorm).

Measured on an RTX 3050 Ti, bf16, activation input 4096 x 3072, forward + backward
(`benchmarks/kernels-phase2.json`, reproduce with `python tools/bench_fusion.py`):

| Module (model family) | Ops fused | Speed vs eager | Peak memory |
|---|---|---|---|
| `NewGELUActivation` (GPT-2) | 8 | **6.8x** | 144 -> 48 MB |
| `FastGELUActivation` | 9 | **6.5x** | 192 -> 48 MB |
| `QuickGELUActivation` (CLIP, others) | 3 | 2.7x | 96 -> 48 MB |
| `ReLUSquaredActivation` | 2 | 2.1x | 96 -> 48 MB |

These are the activation alone, not whole-model speed-ups; the share of a real model's step time they make up is
small for large models and larger for small ones. `--fuse` is new and has been exercised on a tiny GPT-2
(outputs and gradients match, fallbacks tested) but not yet in a full training run.

## Write your own kernel

```bash
tinyforge kernels new my_kernel                          # writes tinyforge_kernels/my_kernel.py (a working GeGLU example)
tinyforge kernels verify --file tinyforge_kernels/my_kernel.py   # same outputs AND gradients as PyTorch?
tinyforge ft train ... --kernels auto --kernel-path tinyforge_kernels/my_kernel.py
tinyforge serve --kernels auto --kernel-path tinyforge_kernels/my_kernel.py
```

A kernel file defines `SPEC` (or `SPECS`): a `KernelSpec` with the PyTorch `reference`, your Triton `kernel`,
`tunables`, `make_inputs`, an `example_shape`, and (to be used in real runs) a `Target` saying which modules it
replaces (`match`, `shape_for`, `make_forward`). The template documents every field. It goes through exactly the
same gate as the built-in kernels: verified, tuned, used only if faster (or `memory_kernel=True`), restored on any
failure, covered by the drift guard. Results appear in `kernel_report.json` as `user:NAME`.

Safety: a kernel file is your own code running with your privileges. tinyforge reads only paths you pass with
`--kernel-path` or `TINYFORGE_KERNEL_PATH`; it never loads anything from the current directory, from a JobSpec, or
from the network. "Verified" means it matched PyTorch on the shapes and dtypes `verify` ran, not in general.

## What is and is not verified

* Verified by tests (GPU: RTX 3050 Ti 4 GB, torch 2.6.0+cu124, triton 3.8.0, Windows): output and gradient
  equivalence in fp32/fp16/bf16 on non-power-of-two widths (960, 1536, ...), GQA rotary, cos/sin shared across the
  batch, all-ignored labels in the loss, LoRA-wrapped MLPs, gradient-checkpointing recompute, fallback on injected
  failure, a deliberately wrong kernel being rejected.
* `kernels verify` checks the shapes and dtype it prints, nothing more.
* **Not verified:** Linux, any GPU other than the 3050 Ti, multi-GPU (DDP) with kernels, 4-bit (QLoRA) models with
  kernels (norm weights become fp32 under `prepare_model_for_kbit_training` while activations stay bf16, so
  `rmsnorm` falls back there), models other than Qwen2/Llama/Mistral-style, long training runs (the acceptance run
  is 30 steps; bf16 loss curves drift slightly even with identical maths, so the comparison uses a 5% tolerance).
* Speed-up numbers come only from `benchmarks/`; do not quote any other number.

## Known limits and caveats

* The RoPE patch replaces the module-level `apply_rotary_pos_emb` function of the model's file, so it affects every
  model of that family in the same process until `restore_kernels(model)` is called.
* The backward of `rmsnorm` always computes the weight gradient even though norm weights are frozen under LoRA
  (a small waste, not a correctness problem).
* Greedy generation with kernels on can differ from kernels off by a token here and there: bf16 maths is not
  associative, and the kernels round in slightly different places. The logits agree within tolerance.
* Triton needs a working CUDA GPU; without one everything silently stays on PyTorch (`kernels doctor` says why).

## Results

Everything below is from one machine (Windows 11, RTX 3050 Ti Laptop 4 GB, torch 2.6.0+cu124, triton 3.8.0) and
lives in `benchmarks/`. Each training comparison trains the same short run twice in separate processes (kernels
`off`, then `auto`); `kernel_report.json` at the end of each run confirms no kernel fell back.

| Run | tokens/s | peak GPU memory | final loss |
|---|---|---|---|
| SmolLM2-360M, bf16, 30 steps: off | 1406 | 2.19 GB | 1.39132 |
| SmolLM2-360M, bf16, 30 steps: auto | 1611 (**+14.5%**) | 1.34 GB (**-39%**) | 1.39132 |
| Qwen2.5-1.5B, 4-bit QLoRA, 20 steps: off | 421 | 2.95 GB | 0.06290 |
| Qwen2.5-1.5B, 4-bit QLoRA, 20 steps: auto | 439 (**+4.3%**) | 2.50 GB (**-15%**) | 0.06276 |

Files: `kernels-phase1-smollm2-360m.json`, `kernels-phase1-qwen1p5b-4bit.json`. Two earlier 360M runs gave +17.4%
and +14.5%, so read the speed-up as "about 15%, give or take a couple of points". The memory saving is the more
reliable number and comes mostly from the fused cross-entropy. The QLoRA gain is smaller because 4-bit matmuls
dominate the step time and `rmsnorm` steps aside there (the weights are fp32 after QLoRA preparation).

**Generation.** With kernels on, greedy generation is not token-for-token identical to kernels off: over 80 new
tokens, 4 of 8 prompts were identical and the others diverged at token 6, 15, 28 and 43, with a largest first-token
logit difference of about 0.2-0.3 on logits of order 10 (`kernels-generation-smollm2-360m.json`, reproduce with
`python tools/bench_generation.py RUN_DIR OUT.json`). That is the size of ordinary bf16 rounding differences, but
it means output text can differ slightly; use `--kernels off` where bit-for-bit reproducibility matters.

**Not run:** Linux, any other GPU, multi-GPU, a full-length fine-tune with `--fuse`.
