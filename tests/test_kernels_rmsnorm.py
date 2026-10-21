import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tinyforge.kernels import registry
from tinyforge.kernels.verify import verify

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@cuda
@pytest.mark.parametrize("shape", [(1, 96), (7, 960), (64, 1536), (33, 2048)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_equivalence(shape, dtype):
    from tinyforge.kernels import rmsnorm  # noqa: F401  (registers the spec)

    r = verify(registry.get("rmsnorm"), shape, dtype, device="cuda", num_warps=4)
    assert r.ok, r.reason


@cuda
def test_weight_grad_matches():
    from tinyforge.kernels.rmsnorm import rmsnorm_fn, rmsnorm_reference

    torch.manual_seed(0)
    x = torch.randn(40, 960, device="cuda", dtype=torch.bfloat16)
    w = (1 + 0.1 * torch.randn(960, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    g = torch.randn(40, 960, device="cuda", dtype=torch.bfloat16)
    w2 = w.detach().clone().requires_grad_()
    (rmsnorm_fn(x, w) * g).sum().backward()
    (rmsnorm_reference(x, w2) * g).sum().backward()
    scale = max(1.0, w2.grad.float().abs().max().item())
    assert (w.grad.float() - w2.grad.float()).abs().max() < 0.05 * scale


@cuda
def test_checkpoint_recompute_identical():
    from tinyforge.kernels.rmsnorm import rmsnorm_fn

    torch.manual_seed(0)
    w = torch.ones(960, device="cuda", dtype=torch.float16, requires_grad=True)
    x1 = torch.randn(16, 960, device="cuda", dtype=torch.float16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    (rmsnorm_fn(x1, w) ** 2).sum().backward()
    g_plain = x1.grad.clone()
    (checkpoint(rmsnorm_fn, x2, w, use_reentrant=False) ** 2).sum().backward()
    assert torch.equal(g_plain, x2.grad)
