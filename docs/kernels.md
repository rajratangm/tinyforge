# Kernels: verified GPU speed-ups for training and serving

A **kernel** is a small program that runs one step of the maths on the GPU. A model does the same few steps
millions of times (normalise, activate, rotate, compute the loss), so a leaner version of those steps makes
training faster and uses less GPU memory without changing the answer.

tinyforge ships Triton kernels for the steps below and **only uses one if it proves itself first**:

1. it must give the same output **and the same gradients** as the plain PyTorch step (the *verifier*),
2. in `auto` mode it must also be faster, or save memory (the *autotuner*),
3. if anything goes wrong while running, the original PyTorch step is put back and the run continues.

Status: **opt-in, default off** (`--kernels off`). The default stays off until the acceptance runs in
`benchmarks/` say otherwise. See "What is and is not verified" before relying on it.

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

Recorded runs live in `benchmarks/`.

**Fine-tuning SmolLM2-360M-Instruct, 30 steps, bf16, 256-token examples, RTX 3050 Ti 4 GB, kernels off vs auto**
(`benchmarks/kernels-phase1-smollm2-360m.json`; each mode in its own process; `kernel_report.json` written at the
end of the run shows all four kernels still active, nothing fell back):

| | off | auto | change |
|---|---|---|---|
| tokens/s (median, step 1 excluded) | 1406 | 1611 | **+14.5%** |
| peak GPU memory | 2.19 GB | 1.34 GB | **-0.84 GB (-39%)** |
| final training loss | 1.39132 | 1.39132 | same |

Two valid runs on this machine gave +17.4% and +14.5%, so read it as "about +15%, run to run noise of a couple of
points". One model, one GPU, 30 steps: indicative, not a guarantee for other setups. Per-kernel micro-benchmark
speed-ups at the tuned shapes (reference time / kernel time, includes backward): rmsnorm 2.7x, rope 1.9x,
fused cross-entropy 1.5x, swiglu 1.3x. Most of the memory saving comes from the fused cross-entropy.

Also recorded: `kernels-smoke.json` (Triton compiles and runs on the 3050 Ti).
Not yet run: Qwen2.5-1.5B 4-bit; the greedy-generation comparison is partly done (see the handoff file).
