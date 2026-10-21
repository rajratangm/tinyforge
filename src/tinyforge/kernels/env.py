"""Can Triton kernels run here? One tiny real kernel answers it; nothing in here raises."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelEnv:
    ok: bool
    reason: str = ""
    gpu: str = ""
    compute_capability: str = ""
    triton_version: str = ""


def _smoke() -> None:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _add(a_ptr, b_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(a_ptr + offs, mask=m) + tl.load(b_ptr + offs, mask=m), mask=m)

    a = torch.randn(1000, device="cuda")
    b = torch.randn(1000, device="cuda")
    y = torch.empty_like(a)
    _add[(triton.cdiv(1000, 256),)](a, b, y, 1000, BLOCK=256)
    torch.cuda.synchronize()
    if not torch.allclose(y, a + b):
        raise RuntimeError("smoke kernel produced wrong values")


def triton_status() -> KernelEnv:
    try:
        import torch

        if not torch.cuda.is_available():
            return KernelEnv(False, "CUDA is not available")
        import triton

        gpu = torch.cuda.get_device_name(0)
        cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
        try:
            _smoke()
        except Exception as e:
            return KernelEnv(False, f"Triton smoke kernel failed: {type(e).__name__}: {e}"[:400],
                             gpu, cc, triton.__version__)
        return KernelEnv(True, "", gpu, cc, triton.__version__)
    except Exception as e:  # missing triton, broken torch, ...
        return KernelEnv(False, f"{type(e).__name__}: {e}"[:400])
