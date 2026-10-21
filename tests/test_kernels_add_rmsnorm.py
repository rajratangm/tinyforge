import pytest
import torch

from tinyforge.kernels import registry
from tinyforge.kernels.verify import verify

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("hidden", [960, 1536, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_add_rmsnorm_equivalence(hidden, dtype):
    from tinyforge.kernels import add_rmsnorm  # noqa: F401  (registers the spec)

    spec = registry.get("add_rmsnorm")
    assert spec.needs_grad is False
    r = verify(spec, (5, hidden), dtype, device="cuda", num_warps=4)
    assert r.ok, r.reason


@cuda
def test_new_residual_is_the_rounded_sum():
    from tinyforge.kernels.add_rmsnorm import add_rmsnorm

    x = torch.randn(3, 960, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(3, 960, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(960, device="cuda", dtype=torch.bfloat16)
    _, new_res = add_rmsnorm(x, res, w, 1e-6)
    assert torch.equal(new_res, x + res)
