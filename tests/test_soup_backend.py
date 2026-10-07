"""Soup backend: config mapping, events, file allow-list, exit codes. Uses a fake `soup`, no GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("jsonschema")

import yaml  # noqa: E402

from tinyforge import ft_data, soup_backend, worker  # noqa: E402

SPEC = """
apiVersion: tinyforge.dev/v1alpha1
kind: TrainingJob
metadata: {name: soup-job}
spec:
  backend: soup
  method: qlora
  model: {base: MODELDIR}
  data: {source: org/data, maxLen: 256}
  hyperparameters: {maxSteps: 10, batchSize: 1, gradAccum: 2, learningRate: 0.0001, loraR: 8, loraAlpha: 16}
"""

FAKE = r'''
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
assert argv[:2] == ["--no-telemetry", "--no-audit-log"], argv
assert os.environ["SOUP_TELEMETRY"] == "0" and os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["PYTHONUNBUFFERED"] == "1"
cfg_path = Path(argv[argv.index("--config") + 1])
import yaml
cfg = yaml.safe_load(cfg_path.read_text())
mode = os.environ.get("FAKE_MODE", "ok")
rows = sum(1 for _ in open(cfg["data"]["train"]))
Path(os.environ["FAKE_RECORD"]).write_text(json.dumps({"cfg": cfg, "rows": rows}))
if mode == "oom":
    print("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
    sys.exit(1)
if mode == "crash":
    print("Traceback: something broke")
    sys.exit(3)
for i in range(1, 5):
    sys.stdout.write("\r 50%|#####     | 5/10 [00:09<00:09, 1.7s/it]")
    print("{'loss': '%s', 'grad_norm': '1.2', 'learning_rate': '0.0001', 'num_tokens': '%d', 'epoch': '%s'}"
          % ("nan" if mode == "nan" else 1.0 / i, 400 * i, i / 4))
print("{'train_runtime': '9.1', 'train_loss': '0.3', 'epoch': '1'}")
out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=True)
(out / "adapter_model.safetensors").write_bytes(b"fake-weights")
(out / "adapter_config.json").write_text("{}")
(out / "tokenizer.json").write_text("{}")
(out / "training_args.bin").write_bytes(b"PICKLE-DO-NOT-LOAD")
(out / "checkpoint-10").mkdir(exist_ok=True)
if mode == "noadapter":
    (out / "adapter_model.safetensors").unlink()
'''


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = tmp_path / "fake_soup.py"
    fake.write_text(FAKE, encoding="utf-8")
    record = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.setattr(soup_backend, "find_soup", lambda: [sys.executable, str(fake)])
    model = tmp_path / "model"
    model.mkdir()
    out = tmp_path / "out"
    (out / "data").mkdir(parents=True)
    (out / "data" / "train.jsonl").write_text(
        "\n".join(json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                           {"role": "assistant", "content": f"a{i}"}]}) for i in range(3)))
    for f in ("val.jsonl", "meta.json"):
        (out / "data" / f).write_text("{}")
    monkeypatch.setattr(ft_data, "prepare", lambda *a, **k: pytest.fail("data should be reused"))
    monkeypatch.setattr(worker, "MODEL_CACHE", tmp_path / "cache")
    spec = tmp_path / "spec.yaml"
    spec.write_text(SPEC.replace("MODELDIR", str(model).replace("\\", "/")), encoding="utf-8")
    return spec, out, record


def events(capsys):
    return [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip().startswith("{")]


def test_success_translates_events_collects_only_allowlisted_files_and_writes_config(env, capsys):
    spec, out, record = env
    assert worker.run_job(spec, out) == worker.EXIT_OK
    ev = events(capsys)
    kinds = [e["event"] for e in ev if e["event"] != "diagnostic"]
    assert kinds[0] == "started" and kinds.count("step") == 4 and kinds[-1] == "finished"
    steps = [e for e in ev if e["event"] == "step"]
    assert [s["step"] for s in steps] == [3, 5, 8, 10] and steps[0]["loss"] == 1.0
    assert all(s["tok_per_s"] is None or 0 <= s["tok_per_s"] < 1e6 for s in steps)  # burst lines: no dt
    best = out / "best"
    assert (best / "adapter_model.safetensors").exists() and (best / "adapter_config.json").exists()
    assert not (best / "training_args.bin").exists() and not (best / "checkpoint-10").exists()
    cfg = json.loads((out / "ft_config.json").read_text())
    assert cfg["quant"] == "4bit" and cfg["lora_r"] == 8 and cfg["max_len"] == 256
    rec = json.loads(record.read_text())
    assert rec["cfg"]["training"]["stream_layers"] is True
    assert rec["cfg"]["training"]["quantization"] == "4bit"
    assert rec["cfg"]["training"]["batch_size"] == 1
    assert rec["cfg"]["training"]["gradient_accumulation_steps"] == 2
    assert rec["cfg"]["data"]["max_length"] == 256 and rec["cfg"]["training"]["epochs"] == 1
    assert rec["rows"] == 10 * 1 * 2  # exact step control: maxSteps * batch * accum rows, cycled from 3


def test_oom_maps_to_exit_4_and_crash_to_exit_crash(env, monkeypatch, capsys):
    spec, out, _ = env
    monkeypatch.setenv("FAKE_MODE", "oom")
    assert worker.run_job(spec, out) == worker.EXIT_FIT
    assert [e for e in events(capsys) if e["event"] == "failed"][-1]["reason"] == "out of memory"
    monkeypatch.setenv("FAKE_MODE", "crash")
    assert worker.run_job(spec, out) == worker.EXIT_CRASH


def test_nan_loss_and_missing_adapter_fail_instead_of_succeeding(env, monkeypatch, capsys):
    spec, out, _ = env
    monkeypatch.setenv("FAKE_MODE", "nan")
    assert worker.run_job(spec, out) == worker.EXIT_CRASH
    monkeypatch.setenv("FAKE_MODE", "noadapter")
    assert worker.run_job(spec, out) == worker.EXIT_CRASH
    assert "adapter_model.safetensors" in [e for e in events(capsys) if e["event"] == "failed"][-1]["reason"]


def test_missing_soup_is_a_clear_environment_error(env, monkeypatch, capsys):
    spec, out, _ = env
    monkeypatch.undo()  # drop the fake
    monkeypatch.setattr(ft_data, "prepare", lambda *a, **k: None)
    monkeypatch.setenv("TINYFORGE_SOUP_BIN", str(spec.parent / "nope" / "soup"))
    monkeypatch.setattr(soup_backend.shutil, "which", lambda *_: None)
    monkeypatch.setattr(worker, "MODEL_CACHE", spec.parent / "cache")
    assert worker.run_job(spec, out) == worker.EXIT_CRASH
    assert "Soup is not installed" in [e for e in events(capsys) if e["event"] == "failed"][-1]["reason"]


def test_backend_soup_needs_a_gpu_and_unknown_backend_is_rejected(env, capsys):
    spec, out, _ = env
    spec.write_text(spec.read_text() + "  resources: {gpus: 0}\n", encoding="utf-8")
    assert worker.run_job(spec, out) == worker.EXIT_SPEC
    spec.write_text(spec.read_text().replace("backend: soup", "backend: magic"), encoding="utf-8")
    assert worker.run_job(spec, out) == worker.EXIT_SPEC


def test_parse_metrics_and_config_builder():
    assert soup_backend.parse_metrics("{'loss': '0.5', 'epoch': '0.5'}") == {"loss": 0.5, "epoch": 0.5}
    assert soup_backend.parse_metrics("Loss: 0.5") is None
    assert soup_backend.parse_metrics("{'loss': 'abc'}") == {}
    bar = " 50%|#####     | 5/10 [00:09<00:09, 1.7s/it]"
    assert soup_backend.parse_metrics(bar + "{'loss': '0.5', 'epoch': '0.5'}") == {"loss": 0.5, "epoch": 0.5}
    doc = yaml.safe_load(SPEC.replace("MODELDIR", "m"))
    cfg = soup_backend.build_config(doc, Path("m"), Path("t.jsonl"), Path("o"))
    assert cfg["training"]["lora"] == {"r": 8, "alpha": 16} and cfg["base"] == "m"
    doc["spec"]["method"] = "lora"
    lora_cfg = soup_backend.build_config(doc, Path("m"), Path("t"), Path("o"))
    assert lora_cfg["training"]["quantization"] == "none"


def test_works_with_relative_paths_because_child_runs_in_another_cwd(env, monkeypatch, capsys):
    spec, out, _ = env
    monkeypatch.chdir(out.parent)
    assert worker.run_job(spec, Path("out")) == worker.EXIT_OK
