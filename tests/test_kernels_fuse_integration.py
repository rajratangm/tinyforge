"""Auto-fusion in the patcher (phase 2): an op chain no hand-written kernel covers is generated, verified,
tuned, patched in, and falls back safely."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from tinyforge.kernels import codegen  # noqa: E402
from tinyforge.kernels.patch import apply_kernels, restore_kernels  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYFORGE_CACHE", str(tmp_path))


def gpt2(dtype=torch.float32) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=128, n_embd=64, n_layer=2, n_head=4, n_positions=64, n_inner=256,
                     activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(cfg).to("cuda", dtype).eval()


def ids():
    return torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(1)).to("cuda")


def fused_keys(report) -> list[str]:
    return [k for k in report.patched if k.startswith("fused:")]


def test_fusion_is_off_unless_asked_for():
    from tinyforge.kernels.patch import apply_kernels as apply

    assert "fuse" in apply.__code__.co_varnames  # an explicit opt-in parameter, default False


@cuda
def test_an_unknown_activation_chain_is_generated_adopted_and_matches():
    ref, m = gpt2(), gpt2()
    r = apply_kernels(m, mode="force", fuse=True)
    assert fused_keys(r), r  # GPT-2's NewGELUActivation is a long elementwise chain no library kernel covers
    x = ids()
    a, b = ref(x).logits, m(x).logits
    assert (a - b).abs().max() < 1e-3 * max(1.0, a.abs().max().item())
    assert fused_keys(r), f"the fused kernel fell back at runtime: {r.skipped}"
    ref.train(), m.train()
    ref(x).logits.sum().backward()
    m(x).logits.sum().backward()
    ga, gb = ref.transformer.wte.weight.grad, m.transformer.wte.weight.grad
    assert (ga - gb).abs().max() < 1e-3 * max(1.0, ga.abs().max().item())


@cuda
def test_restore_puts_the_original_activation_back():
    m = gpt2()
    apply_kernels(m, mode="force", fuse=True)
    assert any("forward" in mod.__dict__ for mod in m.modules())
    restore_kernels(m)
    assert not any("forward" in mod.__dict__ for mod in m.modules())


@cuda
def test_a_bad_generated_kernel_is_rejected_and_the_model_is_unchanged(monkeypatch):
    real = codegen.OP_TABLE["tanh"]
    monkeypatch.setitem(codegen.OP_TABLE, "tanh", codegen.OpDef(
        real.arity, real.fwd, lambda g, a, o: [f"({g}) * 0.5"], real.torch_fn, real.positive))
    ref, m = gpt2(), gpt2()
    r = apply_kernels(m, mode="force", fuse=True)
    assert fused_keys(r) == []
    assert any(k.startswith("fused:") and "verif" in why for k, why in r.skipped.items()), r
    x = ids()
    assert (ref(x).logits - m(x).logits).abs().max() < 1e-5


@cuda
def test_fusion_never_runs_on_cpu_inputs_it_was_not_built_for():
    m = gpt2()
    apply_kernels(m, mode="force", fuse=True)
    act = next(mod for mod in m.modules() if type(mod).__name__ == "NewGELUActivation")
    x = torch.randn(3, 5)  # CPU input: the fused forward must step aside, not crash or move data
    assert torch.allclose(act(x), torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-5)
