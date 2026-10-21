"""Rotary embedding (HF half-rotation layout) forward+backward; q and k may differ in head count (GQA)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import registry


@triton.jit
def _rope(x_ptr, cos_ptr, sin_ptr, out_ptr, H, S, D, HALF, sxb, sxh, sxs,
          BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    si = pid % S
    hi = (pid // S) % H
    bi = pid // (S * H)
    offs = tl.arange(0, BLOCK)
    m = offs < HALF
    xb = x_ptr + bi * sxb + hi * sxh + si * sxs
    cb = (bi * S + si) * D
    x1 = tl.load(xb + offs, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(xb + HALF + offs, mask=m, other=0.0).to(tl.float32)
    c1 = tl.load(cos_ptr + cb + offs, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + cb + HALF + offs, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + cb + offs, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + cb + HALF + offs, mask=m, other=0.0).to(tl.float32)
    if BACKWARD:  # x holds the incoming gradient: apply the transpose of the rotation
        o1 = x1 * c1 + x2 * s2
        o2 = x2 * c2 - x1 * s1
    else:
        o1 = x1 * c1 - x2 * s1
        o2 = x2 * c2 + x1 * s2
    ob = out_ptr + ((bi * H + hi) * S + si) * D
    tl.store(ob + offs, o1.to(out_ptr.dtype.element_ty), mask=m)
    tl.store(ob + HALF + offs, o2.to(out_ptr.dtype.element_ty), mask=m)


def _apply(x, cos, sin, backward, num_warps):
    if x.stride(3) != 1:
        x = x.contiguous()
    b, h, s, d = x.shape
    out = torch.empty(b, h, s, d, dtype=x.dtype, device=x.device)
    _rope[(b * h * s,)](x, cos, sin, out, h, s, d, d // 2, x.stride(0), x.stride(1), x.stride(2),
                        BACKWARD=backward, BLOCK=triton.next_power_of_2(d // 2), num_warps=num_warps)
    return out


class _RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, num_warps):
        cos, sin = cos.contiguous(), sin.contiguous()
        ctx.save_for_backward(cos, sin)
        ctx.num_warps = num_warps
        return _apply(q, cos, sin, False, num_warps), _apply(k, cos, sin, False, num_warps)

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        nw = ctx.num_warps
        return _apply(dq, cos, sin, True, nw), _apply(dk, cos, sin, True, nw), None, None, None


def rope_fn(q, k, cos, sin, unsqueeze_dim: int = 1, num_warps: int = 4):
    """Same contract as HF apply_rotary_pos_emb for q [b, hq, s, d], k [b, hk, s, d], cos/sin [b, s, d]."""
    if unsqueeze_dim != 1:
        raise ValueError("rope kernel supports the [batch, heads, seq, dim] layout only (unsqueeze_dim=1)")
    return _RopeFn.apply(q, k, cos, sin, num_warps)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_reference(q, k, cos, sin, unsqueeze_dim: int = 1):
    """HF apply_rotary_pos_emb, copied so the kernel is checked against real model behaviour."""
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _make_inputs(shape, dtype, device):
    b, hq, hk, s, d = shape
    ang = torch.randn(b, s, d // 2, device=device)
    cos = torch.cat([ang.cos()] * 2, -1).to(dtype)
    sin = torch.cat([ang.sin()] * 2, -1).to(dtype)
    return (torch.randn(b, hq, s, d, device=device, dtype=dtype),
            torch.randn(b, hk, s, d, device=device, dtype=dtype), cos, sin)


registry.register(registry.KernelSpec(
    name="rope",
    # cos/sin are fixed tables, so they take no gradient on either side
    reference=lambda q, k, cos, sin: rope_reference(q, k, cos.detach(), sin.detach()),
    kernel=lambda q, k, cos, sin, num_warps=4: rope_fn(q, k, cos.detach(), sin.detach(), 1, num_warps),
    dtypes=(torch.float16, torch.bfloat16, torch.float32),
    supports=lambda shape: len(shape) == 5 and shape[-1] % 2 == 0,
    tunables={"num_warps": [1, 2, 4]},
    make_inputs=_make_inputs,
))
