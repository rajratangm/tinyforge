"""Fused RMSNorm forward in Triton (inference path). One program per row, fp32 accumulation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
