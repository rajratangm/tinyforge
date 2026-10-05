"""End-to-end smoke test on synthetic text (no network): data -> train -> eval -> API."""

import random

import pytest
from fastapi.testclient import TestClient

from tinyforge import config, data
from tinyforge.evaluate import evaluate
from tinyforge.train import train


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    rng = random.Random(0)
    words = "the king and queen did speak of love war peace night day".split()
    text = "\n".join(" ".join(rng.choice(words) for _ in range(8)) for _ in range(6000))
    (d / "raw.txt").write_text(text)
    return d


def test_pipeline(workdir):
    data_dir = workdir / "data"
    meta, rep = data.prepare(workdir / "raw.txt", data_dir, vocab_size=300)
    assert meta["train_tokens"] > 10_000
    mc = config.make_model_config("nano", meta["vocab_size"], 64)
    tc = config.TrainConfig(max_steps=60, batch_size=8, grad_accum=1, lr=3e-3, warmup_steps=5,
                            eval_interval=30, ckpt_interval=30, eval_batches=3,
                            data_dir=data_dir, run_dir=workdir / "run")
    events = []
    summary = train(mc, tc, events.append)
    kinds = {e["event"] for e in events}
    assert {"started", "step", "eval", "finished"} <= kinds
    assert summary["best_val_loss"] < 5.0  # uniform over 300 tokens is ~5.7
    assert (workdir / "run" / "best.pt").exists()

    res = evaluate(workdir / "run" / "best.pt", data_dir, workdir / "run" / "eval.json")
    assert res["deterministic"] and res["kv_cache_max_err"] < 1e-3
    assert res["val_loss"] < res["uniform_baseline_loss"]

    # resume continues rather than restarting
    tc2 = tc.model_copy(update={"max_steps": 90})
    events2 = []
    train(mc, tc2, events2.append)
    assert any(e["event"] == "resumed" and e["step"] == 60 for e in events2)


def test_api_basics(monkeypatch):
    from tinyforge.server import app

    monkeypatch.setenv("TINYFORGE_AUTH", "off")  # auth itself is covered in test_server_auth.py
    c = TestClient(app)
    assert c.get("/healthz").json()["status"] == "ok"
    assert "hardware" in c.get("/api/hardware").json()
    assert c.post("/api/generate", json={"run": "../etc"}).status_code == 422
    assert c.post("/api/generate", json={"run": "nonexistent"}).status_code == 404
    assert "tinyforge" in c.get("/").text
