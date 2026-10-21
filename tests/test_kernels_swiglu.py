import pytest
import torch

from tinyforge.kernels import registry
from tinyforge.kernels.verify import verify

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("width", [256, 8960, 8193])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu_equivalence(width, dtype):
    from tinyforge.kernels import swiglu  # noqa: F401  (registers the spec)

    r = verify(registry.get("swiglu"), (33, width), dtype, device="cuda", BLOCK=1024, num_warps=4)
    assert r.ok, r.reason


@cuda
def test_swiglu_is_marked_memory_kernel_and_saves_activation_memory():
    from tinyforge.kernels.swiglu import swiglu_fn, swiglu_reference

    assert registry.get("swiglu").memory_kernel is True

    def peak(fn):
        torch.cuda.empty_cache()
        a = torch.randn(512, 8960, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn(512, 8960, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = fn(a, b)
        held = torch.cuda.memory_allocated() - base
        out.sum().backward()
        return held

    assert peak(swiglu_fn) < peak(swiglu_reference)
