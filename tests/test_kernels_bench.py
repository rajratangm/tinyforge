import pytest
import torch
from typer.testing import CliRunner

from tinyforge.cli import app
from tinyforge.kernels import bench, registry
from tinyforge.kernels.registry import KernelSpec


def _run(tok_s, peak, losses):
    return {"tok_per_s": tok_s, "peak_mem_gb": peak, "losses": losses, "final_loss": losses[-1],
            "kernel_report": {"patched": ["rmsnorm"], "skipped": {}, "speedups": {}}}


def _stub(off, auto):
    return lambda mode: off if mode == "off" else auto


def test_bench_result_schema_and_derived_numbers():
    r = bench.bench_model("org/m", steps=10, runner=_stub(_run(1000, 2.0, [3.0, 2.0, 1.5]),
                                                          _run(1200, 1.5, [3.0, 2.01, 1.5])))
    keys = {"model", "steps", "off", "auto", "loss_close", "speedup", "vram_saved_gb", "kernel_report"}
    assert keys <= set(r)
    assert r["off"] == {"tok_per_s": 1000, "peak_mem_gb": 2.0, "final_loss": 1.5}
    assert r["speedup"] == pytest.approx(1.2)
    assert r["vram_saved_gb"] == pytest.approx(0.5)
    assert r["loss_close"] is True


def test_loss_close_is_false_when_curves_diverge():
    r = bench.bench_model("org/m", steps=10, runner=_stub(_run(1000, 2.0, [3.0, 2.0, 1.5]),
                                                          _run(1000, 2.0, [3.0, 2.0, 1.9])))
    assert r["loss_close"] is False


def test_shapes_for_config_covers_every_builtin_kernel():
    from types import SimpleNamespace

    cfg = SimpleNamespace(hidden_size=960, intermediate_size=2560, num_attention_heads=15,
                          num_key_value_heads=5, vocab_size=49152)
    s = bench.shapes_for_config(cfg)
    assert s["rmsnorm"] == (2048, 960) and s["swiglu"] == (2048, 2560)
    assert s["rope"] == (1, 15, 5, 256, 64)
    assert s["fused_ce"] == (256, 512, 49152)


@pytest.fixture
def toy_specs():
    def make(name, factor):
        return KernelSpec(
            name=name, reference=lambda x: x * 2, kernel=lambda x, **t: x * factor, dtypes=(torch.float32,),
            supports=lambda shape: True, example_shape=(4, 8),
            make_inputs=lambda shape, dtype, device: (torch.randn(shape, dtype=dtype, device=device),))

    registry.register(make("toy_ok", 2.0))
    registry.register(make("toy_wrong", 2.5))
    yield
    for name in ("toy_ok", "toy_wrong"):  # a snapshot restore would also drop built-ins imported meanwhile
        registry._SPECS.pop(name, None)


def test_verify_cli_exit_code_nonzero_on_failure(toy_specs):
    r = CliRunner().invoke(app, ["kernels", "verify", "--only", "toy_wrong", "--device", "cpu",
                                 "--dtype", "fp32"])
    assert r.exit_code == 1
    assert "toy_wrong" in r.output


def test_verify_cli_exit_code_zero_when_all_match(toy_specs):
    args = ["kernels", "verify", "--only", "toy_ok", "--device", "cpu", "--dtype", "fp32"]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output


def test_failed_bench_does_not_leave_a_stale_result_file(tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    out.write_text('{"speedup": 99}', encoding="utf-8")

    def boom(*a, **k):
        raise RuntimeError("training crashed")

    monkeypatch.setattr(bench, "bench_model", boom)
    r = CliRunner().invoke(app, ["kernels", "bench", "--out", str(out)])
    assert r.exit_code != 0
    assert not out.exists()
