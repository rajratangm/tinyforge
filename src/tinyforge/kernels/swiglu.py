"""Fused silu(gate) * up with a backward pass that recomputes silu, so no intermediate is kept for autograd."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from . import registry


@triton.jit
def _swiglu_fwd(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0)
    silu = (g * tl.sigmoid(g)).to(u.dtype)  # HF rounds silu(gate) to the input dtype before the multiply
    tl.store(y_ptr + offs, silu * u, mask=m)


@triton.jit
def _swiglu_bwd(dy_ptr, g_ptr, u_ptr, dg_ptr, du_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    dy = tl.load(dy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    tl.store(dg_ptr + offs, (dy * u * s * (1.0 + g * (1.0 - s))).to(dg_ptr.dtype.element_ty), mask=m)
    tl.store(du_ptr + offs, (dy * g * s).to(du_ptr.dtype.element_ty), mask=m)


class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, block, num_warps):
        gate, up = gate.contiguous(), up.contiguous()
        y = torch.empty_like(gate)
        n = gate.numel()
        _swiglu_fwd[(triton.cdiv(n, block),)](gate, up, y, n, BLOCK=block, num_warps=num_warps)
        ctx.save_for_backward(gate, up)
        ctx.cfg = (block, num_warps)
        return y

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        block, num_warps = ctx.cfg
        dy = dy.contiguous()
        dg, du = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        _swiglu_bwd[(triton.cdiv(n, block),)](dy, gate, up, dg, du, n, BLOCK=block, num_warps=num_warps)
        return dg, du, None, None


def swiglu_fn(gate: torch.Tensor, up: torch.Tensor, BLOCK: int = 1024, num_warps: int = 4) -> torch.Tensor:
    return _SwiGLUFn.apply(gate, up, BLOCK, num_warps)


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def _make_inputs(shape, dtype, device):
    return torch.randn(shape, device=device, dtype=dtype), torch.randn(shape, device=device, dtype=dtype)


registry.register(registry.KernelSpec(
    name="swiglu",
    reference=swiglu_reference,
    kernel=lambda gate, up, BLOCK=1024, num_warps=4: swiglu_fn(gate, up, BLOCK, num_warps),
    dtypes=(torch.float16, torch.bfloat16, torch.float32),
    supports=lambda shape: len(shape) >= 1,
    tunables={"BLOCK": [512, 1024, 2048], "num_warps": [4, 8]},
    make_inputs=_make_inputs,
    memory_kernel=True,
))
