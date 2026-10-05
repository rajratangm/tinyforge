"""Worker contract: spec -> FTConfig mapping, validation exit codes, exit-code mapping. No real training."""

from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("jsonschema")

from typer.testing import CliRunner  # noqa: E402

from tinyforge import finetune, ft_data, worker  # noqa: E402
from tinyforge.cli import app  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "spec" / "examples" / "sql-finetune.yaml"
MINIMAL = """
apiVersion: tinyforge.dev/v1alpha1
kind: TrainingJob
metadata: {name: t1}
spec:
  method: lora
  model: {base: org/model}
  data: {source: org/data}
"""


def _write(tmp_path, text: str) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def _events(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_example_maps_to_ftconfig(tmp_path):
    cfg = worker.to_ft_config(worker.load_spec(EXAMPLE), tmp_path)
    assert cfg.base_model == "HuggingFaceTB/SmolLM2-360M"
    assert (cfg.max_steps, cfg.max_len, cfg.eval_interval, cfg.quant) == (240, 384, 50, "none")
    assert cfg.run_dir == tmp_path and cfg.data_dir == tmp_path / "data"


def test_qlora_maps_to_4bit_and_hyperparameters(tmp_path):
    p = _write(tmp_path, MINIMAL.replace("lora\n", "qlora\n") +
               "  hyperparameters: {maxSteps: 7, learningRate: 0.001, loraR: 8, seed: 3}\n"
               "  checkpoint: {everySteps: 2}\n")
    cfg = worker.to_ft_config(worker.load_spec(p), tmp_path)
    got = (cfg.quant, cfg.max_steps, cfg.lr, cfg.lora_r, cfg.seed, cfg.eval_interval)
    assert got == ("4bit", 7, 1e-3, 8, 3, 2)


@pytest.mark.parametrize("extra", [
    "  bogus: 1\n",
    "  resources: {gpus: -1}\n",
    "  hyperparameters: {maxSteps: 0}\n",
])
def test_invalid_spec_exits_2(tmp_path, capsys, extra):
    assert worker.run_job(_write(tmp_path, MINIMAL + extra), tmp_path / "o") == worker.EXIT_SPEC
    assert _events(capsys)[-1]["event"] == "failed"


@pytest.mark.parametrize("extra", ["  resources: {gpus: 2}\n", "  resources: {nodes: 2}\n"])
def test_unsupported_scale_exits_2(tmp_path, extra):
    assert worker.run_job(_write(tmp_path, MINIMAL + extra), tmp_path / "o") == worker.EXIT_SPEC


def test_method_full_exits_2(tmp_path):
    spec = _write(tmp_path, MINIMAL.replace("lora\n", "full\n"))
    assert worker.run_job(spec, tmp_path / "o") == worker.EXIT_SPEC


def test_missing_file_exits_2(tmp_path):
    assert worker.run_job(tmp_path / "nope.yaml", tmp_path / "o") == worker.EXIT_SPEC


def test_dry_run_prints_config_and_trains_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(finetune, "train", lambda *a, **k: pytest.fail("dry-run must not train"))
    r = CliRunner().invoke(app, ["worker", "run", "--spec", str(EXAMPLE), "--out", str(tmp_path / "o"),
                                 "--dry-run"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout.strip().splitlines()[-1])["max_steps"] == 240
    assert not (tmp_path / "o").exists()


@pytest.fixture
def ready(tmp_path, monkeypatch):
    """Spec file + pre-populated data dir so run_job skips data prep."""
    out = tmp_path / "out"
    (out / "data").mkdir(parents=True)
    for f in ("train.jsonl", "val.jsonl", "meta.json"):
        (out / "data" / f).write_text("{}")
    monkeypatch.setattr(ft_data, "prepare", lambda *a, **k: pytest.fail("data should be reused"))
    return _write(tmp_path, MINIMAL + "  gates: [{metric: val_loss, op: '<', value: 1.0}]\n"), out


def _fake_train(summary=None, fail: str | None = None, raise_exc: BaseException | None = None):
    def train(cfg, on_event):
        if fail:
            on_event({"event": "failed", "t": 0, "reason": fail})
        if raise_exc:
            raise raise_exc
        on_event({"event": "finished", "t": 0, **(summary or {})})
        return summary
    return train


def test_success_passes_gate(ready, monkeypatch, capsys):
    spec, out = ready
    monkeypatch.setattr(finetune, "train", _fake_train({"best_val_loss": 0.5}))
    assert worker.run_job(spec, out) == worker.EXIT_OK
    ev = _events(capsys)
    assert [e for e in ev if e["event"] == "gate"][0]["passed"] is True


def test_failed_gate_exits_3(ready, monkeypatch, capsys):
    spec, out = ready
    monkeypatch.setattr(finetune, "train", _fake_train({"best_val_loss": 1.5}))
    assert worker.run_job(spec, out) == worker.EXIT_GATE
    assert _events(capsys)[-1]["reason"] == "gate failed"


def test_unevaluable_gate_fails_closed(ready, monkeypatch, tmp_path):
    _, out = ready
    spec = _write(tmp_path, MINIMAL + "  gates: [{metric: task_exact_match_gain_pts, op: '>=', value: 1}]\n")
    monkeypatch.setattr(finetune, "train", _fake_train({"best_val_loss": 0.1}))
    assert worker.run_job(spec, out) == worker.EXIT_GATE  # data.format defaults to text: no task evaluator


@pytest.mark.parametrize("reason,code", [("out of memory", worker.EXIT_FIT),
                                         ("plan did not fit hardware", worker.EXIT_FIT),
                                         ("training diverged", worker.EXIT_DIVERGED)])
def test_train_failures_map_to_exit_codes(ready, monkeypatch, reason, code):
    spec, out = ready
    monkeypatch.setattr(finetune, "train", _fake_train(fail=reason, raise_exc=RuntimeError(reason)))
    assert worker.run_job(spec, out) == code


def test_unknown_runtime_error_is_a_crash(ready, monkeypatch):
    spec, out = ready
    monkeypatch.setattr(finetune, "train", _fake_train(raise_exc=RuntimeError("boom")))
    assert worker.run_job(spec, out) == worker.EXIT_CRASH


def test_sigterm_exits_75(ready, monkeypatch, capsys):
    spec, out = ready
    old = signal.getsignal(signal.SIGTERM)

    def train(cfg, on_event):
        signal.raise_signal(signal.SIGTERM)  # the handler installed by run_job raises Preempted
        pytest.fail("not reached")

    monkeypatch.setattr(finetune, "train", train)
    try:
        assert worker.run_job(spec, out) == worker.EXIT_PREEMPTED
    finally:
        signal.signal(signal.SIGTERM, old)
    assert _events(capsys)[-1]["reason"] == "preempted"
