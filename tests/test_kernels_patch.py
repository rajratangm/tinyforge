import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from tinyforge.kernels import patch as kpatch
from tinyforge.kernels.env import KernelEnv
from tinyforge.kernels.patch import KernelReport, apply_kernels, restore_kernels

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _tune_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYFORGE_CACHE", str(tmp_path))
    import transformers.models.qwen2.modeling_qwen2 as q2

    orig = q2.apply_rotary_pos_emb  # rope is patched on the module, so tests that never restore would leak it
    yield
    q2.apply_rotary_pos_emb = orig


def tiny_cfg() -> Qwen2Config:
    return Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)


def tiny(dtype=torch.float32, device="cpu") -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    return Qwen2ForCausalLM(tiny_cfg()).to(device=device, dtype=dtype)


def ids(device="cpu"):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 128, (3, 24), generator=g).to(device)


def patched_forwards(model) -> int:
    return sum("forward" in m.__dict__ for m in model.modules())


def test_off_changes_nothing():
    m = tiny()
    r = apply_kernels(m, mode="off")
    assert r == KernelReport()
    assert patched_forwards(m) == 0


def test_no_triton_returns_all_skipped_no_exception(monkeypatch):
    monkeypatch.setattr(kpatch, "triton_status", lambda: KernelEnv(False, "no triton here"))
    m = tiny()
    r = apply_kernels(m, mode="auto")
    assert r.patched == []
    assert r.skipped and all("no triton here" in why for why in r.skipped.values())
    assert patched_forwards(m) == 0


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_patched_model_outputs_and_grads_match(dtype):
    ref, m = tiny(dtype, "cuda"), tiny(dtype, "cuda")
    r = apply_kernels(m, mode="force")
    assert {"rmsnorm", "swiglu", "rope"} <= set(r.patched), r
    x = ids("cuda")
    lo_ref, lo = ref(x).logits.float(), m(x).logits.float()
    scale = max(1.0, lo_ref.abs().max().item())
    tol = (1e-3 if dtype == torch.float32 else 5e-2) * scale
    assert (lo_ref - lo).abs().max() < tol
    lo_ref.sum().backward()
    lo.sum().backward()
    g_ref, g = ref.model.embed_tokens.weight.grad.float(), m.model.embed_tokens.weight.grad.float()
    # bf16 gradients are compared relative to their own size
    g_tol = (1e-3 if dtype == torch.float32 else 5e-2) * max(1.0, g_ref.abs().max().item())
    assert (g_ref - g).abs().max() < g_tol


@cuda
def test_failure_restores_original(monkeypatch):
    from tinyforge.kernels import rmsnorm

    ref, m = tiny(torch.float32, "cuda"), tiny(torch.float32, "cuda")
    r = apply_kernels(m, mode="force")
    assert "rmsnorm" in r.patched

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(rmsnorm, "rmsnorm_fn", boom)
    x = ids("cuda")
    assert (ref(x).logits - m(x).logits).abs().max() < 1e-3  # still correct: fell back mid-forward
    assert "rmsnorm" not in r.patched and "boom" in r.skipped["rmsnorm"]
    assert (ref(x).logits - m(x).logits).abs().max() < 1e-3  # and keeps working afterwards


@cuda
def test_restore_kernels_removes_every_patch():
    m = tiny(torch.float32, "cuda")
    apply_kernels(m, mode="force")
    assert patched_forwards(m) > 0
    restore_kernels(m)
    assert patched_forwards(m) == 0
    assert m._tinyforge_kernel_report.patched == []


@cuda
def test_lora_wrapped_mlp_still_trains():
    from peft import LoraConfig, get_peft_model

    m = get_peft_model(tiny(torch.float32, "cuda"),
                       LoraConfig(r=4, target_modules="all-linear", task_type="CAUSAL_LM"))
    r = apply_kernels(m, mode="force")
    assert "swiglu" in r.patched
    x = ids("cuda")
    m(x).logits.float().pow(2).mean().backward()
    grads = [p.grad for n, p in m.named_parameters() if "lora_B" in n and p.grad is not None]
    assert grads and any(g.abs().max() > 0 for g in grads)


@cuda
def test_gqa_rope_patched_matches():
    import transformers.models.qwen2.modeling_qwen2 as q2

    orig = q2.apply_rotary_pos_emb
    m = tiny(torch.float32, "cuda")
    apply_kernels(m, mode="force")
    assert q2.apply_rotary_pos_emb is not orig
    q = torch.randn(2, 4, 9, 16, device="cuda")
    k = torch.randn(2, 2, 9, 16, device="cuda")
    ang = torch.randn(1, 9, 8, device="cuda")
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    got = q2.apply_rotary_pos_emb(q, k, cos, sin)
    restore_kernels(m)
    want = q2.apply_rotary_pos_emb(q, k, cos, sin)
    assert q2.apply_rotary_pos_emb is orig
    assert all((a - b).abs().max() < 1e-5 for a, b in zip(got, want, strict=True))


@cuda
def test_inference_layer_fusion_matches():
    ref, m = tiny(torch.bfloat16, "cuda").eval(), tiny(torch.bfloat16, "cuda").eval()
    r = apply_kernels(m, mode="force", inference=True)
    assert "add_rmsnorm" in r.patched, r
    x = ids("cuda")
    with torch.no_grad():
        a, b = ref(x).logits.float(), m(x).logits.float()
    assert (a - b).abs().max() < 5e-2 * max(1.0, a.abs().max().item())
    m.train()  # with grad enabled the fused inference path must stand down
    assert m(x).logits.requires_grad


def test_describe_matches_counts_what_would_be_patched():
    from tinyforge.kernels.patch import describe_matches

    assert describe_matches(tiny()) == {"rmsnorm": 5, "swiglu": 2, "rope": 1, "add_rmsnorm": 2}


def test_describe_matches_works_on_meta_models():
    from tinyforge.kernels.patch import describe_matches

    with torch.device("meta"):
        m = Qwen2ForCausalLM(tiny_cfg())
    assert describe_matches(m)["rmsnorm"] == 5


@cuda
@pytest.mark.parametrize("inference", [False, True])
def test_no_kernel_silently_falls_back_during_a_real_forward_and_backward(inference):
    """A fallback keeps results right but hides a broken kernel; every patched kernel must really run."""
    m = tiny(torch.bfloat16, "cuda")
    r = apply_kernels(m, mode="force", inference=inference)
    expected = {"rmsnorm", "swiglu", "rope"} | ({"add_rmsnorm"} if inference else set())
    assert expected <= set(r.patched), r
    x = ids("cuda")
    if inference:
        with torch.no_grad():
            m.eval()(x)
    else:
        m(x).logits.float().sum().backward()
    assert expected <= set(r.patched), f"a kernel fell back at runtime: {r.skipped}"
    assert r.skipped == {}


@cuda
def test_gradient_checkpointing_neither_trips_the_fallback_nor_changes_gradients():
    """Non-reentrant checkpoint recompute stops early by raising a control-flow exception through every
    patched forward; that must propagate untouched, and the gradients must equal the unpatched ones."""
    def build():
        m = tiny(torch.float32, "cuda")
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
        return m.train()

    ref, m = build(), build()
    r = apply_kernels(m, mode="force")
    x = ids("cuda")
    ref(x).logits.sum().backward()
    m(x).logits.sum().backward()
    assert {"rmsnorm", "swiglu", "rope"} <= set(r.patched) and r.skipped == {}, r
    g_ref, g = ref.model.embed_tokens.weight.grad, m.model.embed_tokens.weight.grad
    assert (g_ref - g).abs().max() < 1e-3 * max(1.0, g_ref.abs().max().item())


@cuda
def test_applying_kernels_twice_does_not_stack_wrappers_and_restore_still_clears_everything():
    m = tiny(torch.float32, "cuda")
    first = sorted(apply_kernels(m, mode="force").patched)  # the second call empties the first report
    second = apply_kernels(m, mode="force")  # e.g. a model reloaded into the same object, or a second call
    assert first and first == sorted(second.patched) and second.skipped == {}
    # one wrapper per module, not a wrapper around a wrapper
    ref = tiny(torch.float32, "cuda")
    x = ids("cuda")
    assert (ref(x).logits - m(x).logits).abs().max() < 1e-3
    restore_kernels(m)
    assert patched_forwards(m) == 0
    import transformers.models.qwen2.modeling_qwen2 as q2

    assert q2.apply_rotary_pos_emb.__module__ == "transformers.models.qwen2.modeling_qwen2"
