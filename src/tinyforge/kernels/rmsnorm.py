"""Fused RMSNorm forward in Triton (inference path). One program per row, fp32 accumulation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import registry


@triton.jit
def _rmsnorm_fwd(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + offs, (x * inv * w).to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    x2 = x.contiguous().view(-1, shape[-1])
    y = torch.empty_like(x2)
    n_cols = x2.shape[1]
    _rmsnorm_fwd[(x2.shape[0],)](x2, weight, y, n_cols, eps, BLOCK=triton.next_power_of_2(n_cols))
    return y.view(shape)


# --- training-capable version: HF semantics (normalise in fp32, cast, then multiply the weight) ---

@triton.jit
def _rms_fwd(x_ptr, w_ptr, y_ptr, rstd_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    xn = (x * rstd).to(y_ptr.dtype.element_ty).to(tl.float32)  # the cast HF does before the weight
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + offs, (xn * w).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rms_bwd(x_ptr, w_ptr, g_ptr, rstd_ptr, dx_ptr, dw_part_ptr, n_rows, n_cols, GRID: tl.constexpr,
             BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK], dtype=tl.float32)
    for row in range(pid, n_rows, GRID):
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)
        gw = g * w
        c = tl.sum(gw * x, axis=0) * rstd * rstd / n_cols
        tl.store(dx_ptr + row * n_cols + offs, ((gw - x * c) * rstd).to(dx_ptr.dtype.element_ty), mask=mask)
        dw += g * x * rstd
    tl.store(dw_part_ptr + pid * n_cols + offs, dw, mask=mask)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, num_warps):
        shape = x.shape
        x2 = x.contiguous().view(-1, shape[-1])
        y = torch.empty_like(x2)
        rstd = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(shape[-1])
        _rms_fwd[(x2.shape[0],)](x2, weight, y, rstd, shape[-1], eps, BLOCK=block, num_warps=num_warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.num_warps, ctx.shape = num_warps, shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, g):
        x2, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2.shape
        g2 = g.contiguous().view(n_rows, n_cols)
        dx = torch.empty_like(x2)
        grid = min(n_rows, 128)
        part = torch.empty(grid, n_cols, device=x2.device, dtype=torch.float32)
        _rms_bwd[(grid,)](x2, weight, g2, rstd, dx, part, n_rows, n_cols, GRID=grid,
                          BLOCK=triton.next_power_of_2(n_cols), num_warps=ctx.num_warps)
        return dx.view(ctx.shape), part.sum(0).to(weight.dtype), None, None


def rmsnorm_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, num_warps: int = 4) -> torch.Tensor:
    """Fused RMSNorm with a backward pass. x and weight must share a dtype (the patcher checks)."""
    return _RMSNormFn.apply(x, weight, eps, num_warps)


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """HF Qwen2RMSNorm/LlamaRMSNorm formula, copied so the kernel is checked against real model behaviour."""
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def _make_inputs(shape, dtype, device):
    rows, hidden = shape
    return (torch.randn(rows, hidden, device=device, dtype=dtype),
            (1 + 0.1 * torch.randn(hidden, device=device)).to(dtype))


registry.register(registry.KernelSpec(
    name="rmsnorm",
    reference=rmsnorm_reference,
    kernel=lambda x, w, num_warps=4: rmsnorm_fn(x, w, 1e-6, num_warps),
    dtypes=(torch.float16, torch.bfloat16, torch.float32),
    supports=lambda shape: len(shape) == 2 and shape[1] <= 16384,
    tunables={"num_warps": [2, 4, 8]},
    make_inputs=_make_inputs,
))
