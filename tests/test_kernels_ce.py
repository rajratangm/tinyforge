import pytest
import torch

from tinyforge.kernels import registry
from tinyforge.kernels.verify import verify

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("vocab", [1000, 49152, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ce_equivalence(vocab, dtype):
    from tinyforge.kernels import ce  # noqa: F401  (registers the spec)

    r = verify(registry.get("fused_ce"), (64, 128, vocab), dtype, device="cuda", BLOCK=2048, num_warps=4)
    assert r.ok, r.reason


@cuda
def test_all_ignored_is_zero_and_no_nan():
    from tinyforge.kernels.ce import fused_ce_chunk_sum

    h = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(500, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = torch.full((32,), -100, device="cuda")
    loss = fused_ce_chunk_sum(h, w, None, y)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(h.grad).all() and h.grad.abs().max() == 0
    assert torch.isfinite(w.grad).all() and w.grad.abs().max() == 0


@cuda
def test_frozen_lm_head_gets_no_weight_grad_but_hidden_does():
    from tinyforge.kernels.ce import fused_ce_chunk_sum

    h = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(500, 64, device="cuda", dtype=torch.bfloat16)  # frozen, as under LoRA
    y = torch.randint(0, 500, (32,), device="cuda")
    fused_ce_chunk_sum(h, w, None, y).backward()
    assert h.grad is not None and w.grad is None


@cuda
def test_peak_memory_below_reference():
    from tinyforge.kernels.ce import fused_ce_chunk_sum
    from tinyforge.memory import _chunk_ce_sum

    def peak(fn):
        torch.cuda.empty_cache()
        h = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(151936, 1024, device="cuda", dtype=torch.bfloat16)
        y = torch.randint(0, 151936, (512,), device="cuda")
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn(h, w, None, y).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    assert peak(fused_ce_chunk_sum) < peak(_chunk_ce_sum)


@cuda
def test_chunked_ce_loss_fused_matches_unfused():
    from tinyforge.memory import chunked_ce_loss

    torch.manual_seed(0)
    head = torch.nn.Linear(64, 700, bias=False).cuda().bfloat16()
    head.weight.requires_grad_(False)
    h1 = torch.randn(2, 40, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_()
    labels = torch.randint(0, 700, (2, 40), device="cuda")
    labels[:, :10] = -100
    a = chunked_ce_loss(h1, head, labels, chunk=16)
    b = chunked_ce_loss(h2, head, labels, chunk=16, fused=True)
    a.backward()
    b.backward()
    assert abs(a.item() - b.item()) < 2e-2 * max(1.0, abs(a.item()))
    assert (h1.grad.float() - h2.grad.float()).abs().max() < 2e-2
