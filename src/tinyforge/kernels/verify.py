"""Equivalence gate: a kernel is used only if its outputs and gradients match the PyTorch reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .registry import KernelSpec

TOL = {torch.float32: 1e-5, torch.bfloat16: 2e-2, torch.float16: 5e-3}


@dataclass
class VerifyResult:
    ok: bool
    max_err: float = 0.0
    grad_err: float = 0.0
    reason: str = ""


def _as_tuple(out) -> tuple[torch.Tensor, ...]:
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def _run(fn, inputs, cotangents, tune):
    out = _as_tuple(fn(*inputs, **tune))
    grads: list[torch.Tensor | None] = []
    wrt = [t for t in inputs if t.requires_grad]
    if wrt:
        loss = sum((o.float() * c.float()).sum() for o, c in zip(out, cotangents, strict=True)
                   if o.requires_grad)
        grads = list(torch.autograd.grad(loss, wrt, allow_unused=True)) if torch.is_tensor(loss) else []
    return out, grads


def verify(spec: KernelSpec, shape: tuple[int, ...], dtype: torch.dtype, device: str = "cuda",
           **tune) -> VerifyResult:
    try:
        torch.manual_seed(0)
        base = spec.make_inputs(shape, dtype, device)

        def fresh():
            return tuple(t.detach().clone().requires_grad_(spec.needs_grad and t.is_floating_point())
                         for t in base)

        ref_in, ker_in = fresh(), fresh()
        with torch.set_grad_enabled(spec.needs_grad):
            ref_probe = _as_tuple(spec.reference(*ref_in))
            gen = torch.Generator(device=device).manual_seed(1)
            cot = tuple(torch.randn(o.shape, generator=gen, device=device, dtype=torch.float32)
                        for o in ref_probe)

            ref_out, ref_g = _run(spec.reference, ref_in, cot, {})
            ker_out, ker_g = _run(spec.kernel, ker_in, cot, tune)
    except Exception as e:
        return VerifyResult(False, float("inf"), float("inf"), f"{type(e).__name__}: {e}"[:400])

    tol = TOL.get(dtype, 1e-5)
    max_err = max((a.float() - b.float()).abs().max().item() for a, b in zip(ref_out, ker_out, strict=True))
    out_scale = max(1.0, *(a.float().abs().max().item() for a in ref_out))
    grad_err, grad_scale = 0.0, 1.0
    if len(ref_g) != len(ker_g):
        return VerifyResult(False, max_err, float("inf"), "gradient count differs from reference")
    for a, b in zip(ref_g, ker_g, strict=True):
        if a is None and b is None:
            continue
        if a is None or b is None:
            return VerifyResult(False, max_err, float("inf"), "gradient missing on one side")
        grad_err = max(grad_err, (a.float() - b.float()).abs().max().item())
        grad_scale = max(grad_scale, a.float().abs().max().item())
    bad = [n for n, e, s in (("output", max_err, out_scale), ("gradient", grad_err, grad_scale))
           if not e <= tol * s]
    reason = f"{' and '.join(bad)} differ from reference" if bad else ""
    return VerifyResult(not bad, max_err, grad_err, reason)
