"""Fused residual-add + RMSNorm for decoding (forward only): one pass instead of an add and a norm."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import registry
from .rmsnorm import rmsnorm_reference


@triton.jit
def _add_rms(x_ptr, r_ptr, w_ptr, y_ptr, s_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=m, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_cols + offs, mask=m, other=0.0).to(tl.float32)
    s = (x + r).to(s_ptr.dtype.element_ty)  # the residual stream is stored rounded, like the unfused add
    tl.store(s_ptr + row * n_cols + offs, s, mask=m)
    sf = s.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(sf * sf, axis=0) / n_cols + eps)
    xn = (sf * rstd).to(y_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + offs, (xn * w).to(y_ptr.dtype.element_ty), mask=m)


def add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                num_warps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (rmsnorm(x + residual), x + residual). Forward only; x, residual and weight share a dtype."""
    shape = x.shape
    x2 = x.contiguous().view(-1, shape[-1])
    r2 = residual.contiguous().view(-1, shape[-1])
    y, s = torch.empty_like(x2), torch.empty_like(x2)
    _add_rms[(x2.shape[0],)](x2, r2, weight, y, s, shape[-1], eps,
                             BLOCK=triton.next_power_of_2(shape[-1]), num_warps=num_warps)
    return y.view(shape), s.view(shape)


def _reference(x, residual, weight):
    s = x + residual
    return rmsnorm_reference(s, weight, 1e-6), s


def _make_inputs(shape, dtype, device):
    rows, hidden = shape
    return (torch.randn(rows, hidden, device=device, dtype=dtype),
            torch.randn(rows, hidden, device=device, dtype=dtype),
            (1 + 0.1 * torch.randn(hidden, device=device)).to(dtype))


registry.register(registry.KernelSpec(
    name="add_rmsnorm",
    reference=_reference,
    kernel=lambda x, r, w, num_warps=4: add_rmsnorm(x, r, w, 1e-6, num_warps),
    dtypes=(torch.float16, torch.bfloat16, torch.float32),
    supports=lambda shape: len(shape) == 2 and shape[1] <= 16384,
    tunables={"num_warps": [2, 4, 8]},
    make_inputs=_make_inputs,
    needs_grad=False,
))
