import numpy as np
import pytest
import torch

from tinyforge import config, infer
from tinyforge.data import BinDataset, quality_checks
from tinyforge.model import GPT
from tinyforge.warnings import Level


def tiny_cfg():
    return config.ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32)


def test_param_count_matches_estimate():
    c = tiny_cfg()
    assert GPT(c).num_params() == c.num_params()


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = GPT(tiny_cfg()).eval()
    ids = torch.randint(0, 64, (1, 12))
    full, _ = m(ids)
    caches = [{} for _ in m.blocks]
    m(ids[:, :-1], caches=caches, start_pos=0)
    inc, _ = m(ids[:, -1:], caches=caches, start_pos=11)
    assert (full - inc).abs().max() < 1e-4


def test_model_can_overfit_one_batch():
    torch.manual_seed(0)
    m = GPT(tiny_cfg())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.randint(0, 64, (4, 32))
    y = torch.roll(x, -1, 1)
    first = m(x, y)[1].item()
    for _ in range(60):
        opt.zero_grad()
        loss = m(x, y)[1]
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_generation_is_seeded_and_bounded():
    m = GPT(tiny_cfg()).eval()
    a = list(infer.generate_ids(m, [1, 2, 3], max_new=10, seed=5))
    b = list(infer.generate_ids(m, [1, 2, 3], max_new=10, seed=5))
    assert a == b and len(a) == 10
    long = list(infer.generate_ids(m, [1], max_new=500, seed=1))
    assert len(long) < 32  # never exceeds the context window


def test_int8_close_to_fp32():
    torch.manual_seed(0)
    m = GPT(tiny_cfg()).eval()
    ids = torch.randint(0, 64, (1, 16))
    ref, _ = m(ids)
    q, _ = infer.quantize_int8(m)(ids)
    assert torch.allclose(ref, q, atol=0.05)


def test_planner_shrinks_to_fit_and_errors_when_impossible():
    mc = config.make_model_config("base", 2048, 1024)
    tc = config.TrainConfig(batch_size=32, grad_accum=1)
    t, rep = config.plan(mc, tc, vram_gb=4, bf16=True)
    assert t.grad_checkpointing
    assert any(i.code == "PL004" for i in rep.items) or t.batch_size < 32
    mc_small = config.make_model_config("nano", 2048, 128)
    _, rep2 = config.plan(mc_small, tc, vram_gb=4, bf16=True)
    assert not rep2.has_errors


def test_config_rejects_bad_heads():
    with pytest.raises(ValueError):
        config.ModelConfig(d_model=30, n_head=4)


def test_data_quality_flags_tiny_corpus():
    rep = quality_checks("a\na\na\na\n", n_tokens=10, vocab_size=2048)
    codes = {i.code for i in rep.items}
    assert {"DQ001", "DQ002"} <= codes
    assert all(i.level in (Level.WARN, Level.ERROR) for i in rep.items)


def test_dataset_rejects_too_short(tmp_path):
    p = tmp_path / "x.bin"
    np.arange(10, dtype=np.uint16).tofile(p)
    with pytest.raises(ValueError):
        BinDataset(p, block_size=64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_rmsnorm_matches_torch():
    pytest.importorskip("triton")
    from tinyforge.kernels.rmsnorm import rmsnorm_triton
    from tinyforge.model import RMSNorm

    x = torch.randn(4, 16, 96, device="cuda", dtype=torch.float16)
    n = RMSNorm(96).cuda().half()
    assert (n(x) - rmsnorm_triton(x, n.weight, n.eps)).abs().max() < 5e-3
