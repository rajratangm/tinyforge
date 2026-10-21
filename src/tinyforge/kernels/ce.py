"""Fused cross-entropy over one chunk of positions.

The reference path upcasts the [N, vocab] logits to fp32 and keeps softmax intermediates for autograd.
Here the logits stay in the model dtype and a Triton kernel turns that same buffer into dlogits in place,
so a chunk costs one [N, vocab] buffer in total. The matmuls stay in cuBLAS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from . import registry

IGNORE = -100


@triton.jit
def _ce(logits_ptr, y_ptr, loss_ptr, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * V
    y = tl.load(y_ptr + row)
    valid = y != -100
    offs0 = tl.arange(0, BLOCK)
    m = tl.max(tl.load(base + offs0, mask=offs0 < V, other=float("-inf")).to(tl.float32), axis=0)
    for start in range(BLOCK, V, BLOCK):
        offs = start + offs0
        x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + offs0
        x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
        acc += tl.exp(x - m)
    lse = m + tl.log(tl.sum(acc, axis=0))
    target = tl.load(base + tl.where(valid, y, 0)).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(valid, lse - target, 0.0))
    for start in range(0, V, BLOCK):
        offs = start + offs0
        msk = offs < V
        x = tl.load(base + offs, mask=msk, other=float("-inf")).to(tl.float32)
        g = tl.exp(x - lse) - tl.where(offs == y, 1.0, 0.0)
        tl.store(base + offs, tl.where(valid, g, 0.0).to(logits_ptr.dtype.element_ty), mask=msk)


class _FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, labels, block, num_warps):
        logits = F.linear(hidden, weight, bias).contiguous()
        n, v = logits.shape
        rows = torch.empty(n, device=logits.device, dtype=torch.float32)
        _ce[(n,)](logits, labels, rows, v, BLOCK=block, num_warps=num_warps)
        ctx.save_for_backward(logits, hidden, weight)  # `logits` now holds dlogits
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return rows.sum()

    @staticmethod
    def backward(ctx, g):
        dl, hidden, weight = ctx.saved_tensors
        dl.mul_(g)  # in place: the buffer is only ever consumed once (checkpoint recomputes it)
        # autocast computed the logits in bf16/fp16, but hidden and the LM head can be fp32 (QLoRA preparation
        # casts every weight to fp32): matmul in the logits dtype, return gradients in the inputs dtypes
        cd = dl.dtype
        dh = (dl @ weight.to(cd)).to(hidden.dtype) if ctx.needs_input_grad[0] else None
        dw = (dl.t() @ hidden.to(cd)).to(weight.dtype) if ctx.needs_input_grad[1] else None  # None if frozen
        db = dl.sum(0).to(ctx.bias_dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dh, dw, db, None, None, None


def fused_ce_chunk_sum(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                       labels: torch.Tensor, BLOCK: int = 2048, num_warps: int = 8) -> torch.Tensor:
    """Summed (not mean) cross-entropy over `labels != -100` for one chunk. hidden [N, H], weight [V, H]."""
    return _FusedCE.apply(hidden, weight, bias, labels, BLOCK, num_warps)


def _make_inputs(shape, dtype, device):
    n, h, v = shape
    labels = torch.randint(0, v, (n,), device=device)
    labels[torch.rand(n, device=device) < 0.1] = IGNORE
    return (torch.randn(n, h, device=device, dtype=dtype),
            (0.05 * torch.randn(v, h, device=device)).to(dtype), labels)


def _reference(hidden, weight, labels):
    from ..memory import _chunk_ce_sum

    return _chunk_ce_sum(hidden, weight, None, labels)


registry.register(registry.KernelSpec(
    name="fused_ce",
    reference=_reference,
    kernel=lambda h, w, y, BLOCK=2048, num_warps=8: fused_ce_chunk_sum(h, w, None, y, BLOCK, num_warps),
    dtypes=(torch.float16, torch.bfloat16, torch.float32),
    supports=lambda shape: len(shape) == 3,
    tunables={"BLOCK": [2048, 4096], "num_warps": [4, 8]},
    make_inputs=_make_inputs,
    memory_kernel=True,
))
