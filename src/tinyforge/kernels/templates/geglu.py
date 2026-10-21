"""Example user kernel: GeGLU, gelu_tanh(gate) * up (the MLP activation of Gemma-style models).

Copy this file, change it, and then:

    tinyforge kernels verify --file geglu.py                     # same output AND gradients as PyTorch?
    tinyforge ft train ... --kernels auto --kernel-path geglu.py # use it while training
    tinyforge serve --kernels auto --kernel-path geglu.py        # ... or serving

A kernel is used only if it passes `verify` (on the shapes in `example_shape` and the model's own shape) and,
in `auto` mode, is faster or saves memory. If it ever fails while running, tinyforge puts the PyTorch op back.
This file is your own code and runs with your privileges; tinyforge only loads files you point it at.

A user kernel file defines SPEC (one KernelSpec) or SPECS (a list). The fields:
  reference    plain PyTorch version of the op: the ground truth
  kernel       your Triton version, same positional arguments, plus keyword arguments from `tunables`
  tunables     settings to try, e.g. {"num_warps": [4, 8]}; the fastest verified one is cached per GPU
  make_inputs  (shape, dtype, device) -> tuple of input tensors for verify and tune
  example_shape  what `kernels verify --file` runs on
  target       (optional) which model modules to replace, so the kernel is used in real runs
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tinyforge.kernels.registry import KernelSpec, Target

_C = 0.7978845608028654  # sqrt(2 / pi)


@triton.jit
def _geglu_fwd(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0)
    t = 2.0 * tl.sigmoid(2.0 * 0.7978845608028654 * (g + 0.044715 * g * g * g)) - 1.0  # tanh
    gelu = (0.5 * g * (1.0 + t)).to(u.dtype)
    tl.store(y_ptr + offs, gelu * u, mask=m)


@triton.jit
def _geglu_bwd(dy_ptr, g_ptr, u_ptr, dg_ptr, du_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    dy = tl.load(dy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    z = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    t = 2.0 * tl.sigmoid(2.0 * z) - 1.0
    dgelu = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * g * g)
    tl.store(dg_ptr + offs, (dy * u * dgelu).to(dg_ptr.dtype.element_ty), mask=m)
    tl.store(du_ptr + offs, (dy * 0.5 * g * (1.0 + t)).to(du_ptr.dtype.element_ty), mask=m)


class _GeGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, block, num_warps):
        gate, up = gate.contiguous(), up.contiguous()
        y = torch.empty_like(gate)
        n = gate.numel()
        _geglu_fwd[(triton.cdiv(n, block),)](gate, up, y, n, BLOCK=block, num_warps=num_warps)
        ctx.save_for_backward(gate, up)
        ctx.cfg = (block, num_warps)
        return y

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        block, num_warps = ctx.cfg
        dg, du = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        _geglu_bwd[(triton.cdiv(n, block),)](dy.contiguous(), gate, up, dg, du, n, BLOCK=block,
                                              num_warps=num_warps)
        return dg, du, None, None


def geglu_fn(gate: torch.Tensor, up: torch.Tensor, BLOCK: int = 1024, num_warps: int = 4) -> torch.Tensor:
    return _GeGLU.apply(gate, up, BLOCK, num_warps)


def reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.gelu(gate, approximate="tanh") * up


def _make_inputs(shape, dtype, device):
    return torch.randn(shape, device=device, dtype=dtype), torch.randn(shape, device=device, dtype=dtype)


def _is_geglu_mlp(m: torch.nn.Module) -> bool:
    if not all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj", "act_fn")):
        return False
    act = m.act_fn
    return type(act).__name__ in {"GELUTanh", "PytorchGELUTanh"} or (
        isinstance(act, torch.nn.GELU) and act.approximate == "tanh")


def _make_forward(mlp: torch.nn.Module, params: dict):
    def forward(x):
        gate, up = mlp.gate_proj(x), mlp.up_proj(x)  # the projections (and any LoRA on them) stay as they are
        if not (gate.is_cuda and gate.dtype == up.dtype and gate.shape == up.shape):
            return NotImplemented  # use the original forward for this input
        return mlp.down_proj(geglu_fn(gate, up, params["BLOCK"], params["num_warps"]))
    return forward


SPEC = KernelSpec(
    name="geglu",
    reference=reference,
    kernel=lambda gate, up, BLOCK=1024, num_warps=4: geglu_fn(gate, up, BLOCK, num_warps),
    dtypes=(torch.float16, torch.bfloat16, torch.float32),
    supports=lambda shape: len(shape) >= 1,
    tunables={"BLOCK": [512, 1024, 2048], "num_warps": [4, 8]},
    make_inputs=_make_inputs,
    memory_kernel=True,  # keeps no intermediate for autograd: adopted for the memory even if not faster
    example_shape=(512, 1024),
    target=Target(
        match=_is_geglu_mlp,
        shape_for=lambda mlp: (2048, mlp.config.intermediate_size),
        make_forward=_make_forward,
    ),
)
