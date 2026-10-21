import pytest
import torch

from tinyforge.kernels import registry
from tinyforge.kernels.verify import verify

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("heads", [(14, 2), (8, 8)])
@pytest.mark.parametrize("seq", [1, 33, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rope_equivalence_with_gqa(heads, seq, head_dim, dtype):
    from tinyforge.kernels import rope  # noqa: F401  (registers the spec)

    shape = (2, heads[0], heads[1], seq, head_dim)
    r = verify(registry.get("rope"), shape, dtype, device="cuda", num_warps=4)
    assert r.ok, r.reason


@cuda
def test_rope_accepts_transposed_views_like_hf_produces():
    from tinyforge.kernels.rope import rope_fn, rope_reference

    torch.manual_seed(0)
    b, s, hq, hk, d = 2, 17, 14, 2, 64
    q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2)  # [b, h, s, d], strided
    k = torch.randn(b, s, hk, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    ang = torch.randn(b, s, d // 2, device="cuda")
    cos = torch.cat([ang.cos()] * 2, -1).to(torch.bfloat16)
    sin = torch.cat([ang.sin()] * 2, -1).to(torch.bfloat16)
    qo, ko = rope_fn(q, k, cos, sin)
    qr, kr = rope_reference(q, k, cos, sin)
    assert (qo.float() - qr.float()).abs().max() < 2e-2
    assert (ko.float() - kr.float()).abs().max() < 2e-2
    assert qo.shape == q.shape and ko.shape == k.shape


@cuda
def test_rope_cos_with_batch_one_broadcasts_over_the_batch():
    # HF builds cos/sin from position_ids of shape [1, seq] during training, so cos is [1, seq, dim]
    from tinyforge.kernels.rope import rope_fn, rope_reference

    torch.manual_seed(0)
    b, s, d = 3, 21, 64
    q = torch.randn(b, 4, s, d, device="cuda", requires_grad=True)
    k = torch.randn(b, 2, s, d, device="cuda", requires_grad=True)
    ang = torch.randn(1, s, d // 2, device="cuda")
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    qo, ko = rope_fn(q, k, cos, sin)
    qr, kr = rope_reference(q, k, cos, sin)
    assert (qo - qr).abs().max() < 1e-5 and (ko - kr).abs().max() < 1e-5
    (qo.sum() + ko.sum()).backward()
    gq, gk = q.grad.clone(), k.grad.clone()
    q.grad = k.grad = None
    (qr.sum() + kr.sum()).backward()
    assert (gq - q.grad).abs().max() < 1e-5 and (gk - k.grad).abs().max() < 1e-5


@cuda
def test_rope_rejects_cos_batch_that_cannot_broadcast():
    from tinyforge.kernels.rope import rope_fn

    q = torch.randn(4, 2, 5, 16, device="cuda")
    cs = torch.randn(2, 5, 16, device="cuda")
    with pytest.raises(ValueError):
        rope_fn(q, q.clone(), cs, cs)
