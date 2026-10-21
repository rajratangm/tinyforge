"""--kernels across FTConfig, the JobSpec/worker, the CLI, and the training-time drift guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

pytest.importorskip("yaml")
pytest.importorskip("jsonschema")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from tinyforge import finetune, worker  # noqa: E402
from tinyforge.cli import app  # noqa: E402
from tinyforge.kernels.patch import apply_kernels, kernels_agree  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
MINIMAL = """
apiVersion: tinyforge.dev/v1alpha1
kind: TrainingJob
metadata: {name: t1}
spec:
  method: lora
  model: {base: org/model}
  data: {source: org/data}
"""


@pytest.fixture(autouse=True)
def _tune_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYFORGE_CACHE", str(tmp_path / "cache"))


def _cfg(tmp_path, **kw) -> finetune.FTConfig:
    return finetune.FTConfig(base_model="org/model", run_dir=tmp_path, data_dir=tmp_path, **kw)


def _tiny(device="cuda") -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen2ForCausalLM(cfg).to(device)


def test_ftconfig_kernels_default_off(tmp_path):
    assert _cfg(tmp_path).kernels == "off"
    assert _cfg(tmp_path, kernels="force").kernels == "force"


def test_ftconfig_rejects_unknown_kernels_mode(tmp_path):
    with pytest.raises(ValidationError):
        _cfg(tmp_path, kernels="sometimes")


def test_jobspec_accepts_kernels_field(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(MINIMAL + "  kernels: auto\n", encoding="utf-8")
    assert worker.to_ft_config(worker.load_spec(p), tmp_path).kernels == "auto"


def test_jobspec_rejects_bad_kernels_value(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(MINIMAL + "  kernels: sometimes\n", encoding="utf-8")
    assert worker.run_job(p, tmp_path / "o") == worker.EXIT_SPEC


@pytest.mark.parametrize("args", [["ft", "train", "--help"], ["ft", "generate", "--help"],
                                  ["ft", "pipeline", "--help"], ["serve", "--help"]])
def test_cli_exposes_kernels_flag(args):
    out = CliRunner().invoke(app, args).output
    assert "--kernels" in out


@cuda
def test_kernels_agree_true_for_a_correct_patch_and_false_for_a_wrong_one(monkeypatch):
    from tinyforge.kernels import rmsnorm

    m = _tiny()
    apply_kernels(m, mode="force")
    x = torch.randint(0, 128, (2, 16), device="cuda")

    def loss_fn(use_kernels: bool):
        return m(input_ids=x, labels=x).loss

    assert kernels_agree(m, loss_fn) is True
    real = rmsnorm.rmsnorm_fn
    monkeypatch.setattr(rmsnorm, "rmsnorm_fn", lambda *a, **k: real(*a, **k) * 50)  # wrong, yet "works"
    assert kernels_agree(m, loss_fn) is False


def test_kernels_agree_is_true_when_nothing_is_patched():
    assert kernels_agree(torch.nn.Linear(2, 2), lambda use_kernels: torch.tensor(1.0)) is True


@cuda
def test_enable_kernels_writes_report_and_returns_fused_ce_params(tmp_path):
    events = []
    c = _cfg(tmp_path, kernels="force")
    fused = finetune._enable_kernels(_tiny(), c, "cuda", lambda kind, **kw: events.append((kind, kw)))
    report = json.loads(Path(tmp_path / "kernel_report.json").read_text())
    assert {"rmsnorm", "swiglu", "rope", "fused_ce"} <= set(report["patched"])
    assert isinstance(fused, dict) and "BLOCK" in fused
    assert events and events[0][0] == "kernels"


def test_enable_kernels_off_does_nothing(tmp_path):
    fused = finetune._enable_kernels(torch.nn.Linear(2, 2), _cfg(tmp_path), "cuda", lambda *a, **k: None)
    assert fused is False
    assert not (tmp_path / "kernel_report.json").exists()


@cuda
def test_kernel_report_is_rewritten_with_runtime_fallbacks(tmp_path, monkeypatch):
    from tinyforge.kernels import rmsnorm

    m = _tiny()
    c = _cfg(tmp_path, kernels="force")
    finetune._enable_kernels(m, c, "cuda", lambda *a, **k: None)
    assert "rmsnorm" in json.loads((tmp_path / "kernel_report.json").read_text())["patched"]

    def boom(*a, **k):
        raise RuntimeError("kernel died mid-run")

    monkeypatch.setattr(rmsnorm, "rmsnorm_fn", boom)
    m(input_ids=torch.randint(0, 128, (2, 8), device="cuda"))  # falls back at runtime
    finetune._write_kernel_report(m, c)
    final = json.loads((tmp_path / "kernel_report.json").read_text())
    assert "rmsnorm" not in final["patched"] and "kernel died mid-run" in final["skipped"]["rmsnorm"]
