"""KV-cache math, telemetry buffers, the router callback and the /api/telemetry endpoint."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinyforge import server
from tinyforge.engines import LlamaCppBackend
from tinyforge.openai_api import build_router
from tinyforge.telemetry import Telemetry, kv_bytes_per_token, kv_curve, kv_info_from_hf_config

TOKEN = "tok"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_kv_bytes_per_token_matches_known_models():
    assert kv_bytes_per_token(32, 8, 128) == 131072  # Llama-3.1-8B fp16: 128 KiB per token
    assert kv_bytes_per_token(36, 2, 128) == 36864  # Qwen2.5-3B: grouped-query attention, 2 KV heads
    q8 = kv_bytes_per_token(32, 8, 128, "q8_0", "q8_0")
    assert 0.5 < q8 / 131072 < 0.54  # an 8-bit cache is about half
    assert kv_bytes_per_token(32, 8, 128, "q8_0", "f16") < 131072  # K and V types may differ


def test_kv_info_from_hf_config_defaults_and_failures():
    cfg = {"num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096}
    info = kv_info_from_hf_config(cfg)
    assert info["kv_heads"] == 32 and info["head_dim"] == 128  # no GQA field -> multi-head
    assert kv_info_from_hf_config({**cfg, "num_key_value_heads": 8})["bytes_per_token"] == 131072
    assert kv_info_from_hf_config({"nothing": 1}) is None


def test_kv_curve_is_a_straight_line_through_zero():
    c = kv_curve(131072, 4096, 8)
    assert c[0] == {"tokens": 0, "mb": 0.0} and c[-1]["tokens"] == 4096
    slopes = {round(b["mb"] / b["tokens"], 4) for b in c[1:]}
    assert len(slopes) == 1 and slopes.pop() == pytest.approx(0.125, rel=1e-3)  # 128 KiB per token


def test_telemetry_records_requests_with_kv_use_and_bounds_its_buffers():
    t = Telemetry(max_requests=3)
    t.set_engine("m", "llamacpp", {"bytes_per_token": 131072.0}, ctx_alloc_tokens=4096, max_concurrency=2)
    r = t.record_request(100, 20, 2.0, "m")
    assert r["context_tokens"] == 120 and r["tok_per_s"] == 10.0
    assert r["kv_used_mb"] == pytest.approx(120 * 131072 / 1024**2, abs=0.01)
    for _ in range(5):
        t.record_request(1, 1, 1.0)
    assert len(t.snapshot()["requests"]) == 3  # ring buffer
    snap = t.snapshot()
    assert snap["kv_alloc_mb"] == 512.0 and snap["engine"]["kv_buffer"].startswith("preallocated")
    assert snap["kv_curve"][-1]["tokens"] == 4096


def test_hf_engine_is_described_as_growing_and_unknown_kv_is_handled():
    t = Telemetry()
    t.set_engine("hf", "hf", None)
    snap = t.snapshot()
    assert snap["engine"]["kv_buffer"] == "grows with the tokens in context" and snap["kv_curve"] == []
    assert t.record_request(5, 5, 0.0)["tok_per_s"] is None  # no division by zero


def test_sample_reports_ram_and_never_raises_without_a_gpu(monkeypatch):
    monkeypatch.setattr("tinyforge.telemetry.gpu_memory", lambda: None)
    s = Telemetry().sample()
    assert s["ram_used_gb"] > 0 and "vram_used_gb" not in s


class Fake:
    name = "fake"

    def stream(self, messages, max_tokens, temperature, top_p):
        yield from ("Hel", "lo")

    def count_tokens(self, text):
        return len(text.split()) or 1


def test_router_reports_each_request_in_both_modes_and_survives_a_failing_callback():
    got = []
    app = FastAPI()
    app.include_router(build_router(lambda: None, lambda: Fake(), on_request=got.append))
    c = TestClient(app)
    body = {"messages": [{"role": "user", "content": "one two three"}]}
    assert c.post("/v1/chat/completions", json=body).status_code == 200
    assert c.post("/v1/chat/completions", json={**body, "stream": True}).status_code == 200
    assert [g["stream"] for g in got] == [False, True] and got[0]["prompt_tokens"] == 3
    assert got[0]["completion_tokens"] >= 1 and got[0]["seconds"] >= 0

    def boom(_):
        raise RuntimeError("callback bug")

    app2 = FastAPI()
    app2.include_router(build_router(lambda: None, lambda: Fake(), on_request=boom))
    assert TestClient(app2).post("/v1/chat/completions", json=body).status_code == 200


def test_llamacpp_flags_for_inference_techniques(tmp_path):
    m = tmp_path / "m.gguf"
    m.write_bytes(b"0")
    d = tmp_path / "d.gguf"
    d.write_bytes(b"0")
    base = LlamaCppBackend(m, server_bin=["x"], ngl=1)
    cmd = base.build_cmd(1)
    assert "--flash-attn" in cmd and "--cache-type-k" not in cmd and "--spec-type" not in cmd
    fast = LlamaCppBackend(m, server_bin=["x"], ngl=1, flash_attn="on", kv_type="q8_0", speculative="ngram")
    c = fast.build_cmd(1)
    assert c[c.index("--flash-attn") + 1] == "on" and c[c.index("--cache-type-k") + 1] == "q8_0"
    assert c[c.index("--cache-type-v") + 1] == "q8_0" and c[c.index("--spec-type") + 1] == "ngram-mod"
    dr = LlamaCppBackend(m, server_bin=["x"], ngl=1, speculative="draft", draft_model=d).build_cmd(1)
    assert dr[dr.index("--spec-type") + 1] == "draft-simple" and str(d) in dr
    for bad in ({"flash_attn": "x"}, {"kv_type": "q1"}, {"speculative": "magic"}, {"speculative": "draft"}):
        with pytest.raises(ValueError):
            LlamaCppBackend(m, server_bin=["x"], **bad)


def test_telemetry_endpoint_is_authenticated_and_reflects_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYFORGE_API_TOKEN", TOKEN)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "_backend", Fake())
    server.telemetry.requests.clear()
    c = TestClient(server.app)
    assert c.get("/api/telemetry").status_code == 401
    body = {"messages": [{"role": "user", "content": "a b c"}]}
    assert c.post("/v1/chat/completions", json=body, headers=AUTH).status_code == 200
    snap = c.get("/api/telemetry", headers=AUTH).json()
    assert {"engine", "kv_curve", "samples", "requests"} <= snap.keys()
    assert snap["requests"] and snap["requests"][-1]["prompt_tokens"] == 3
    if server._manager is not None:
        server._manager.stop()
    server._manager = None


def test_dashboard_endpoints_are_authenticated_validated_and_shaped(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYFORGE_API_TOKEN", TOKEN)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    c = TestClient(server.app)
    for path in ("/api/methods/suggest", "/api/bench/suggest", "/api/hardware"):
        assert c.get(path).status_code == 401
    hw = c.get("/api/hardware", headers=AUTH).json()
    assert any(x["name"] == "torch" for x in hw["components"])
    m = c.get("/api/methods/suggest?params_b=1.5", headers=AUTH).json()
    assert {r["name"] for r in m["methods"]} >= {"lora", "qlora", "dpo"} and "free_ram_gb" in m["context"]
    b = c.get("/api/bench/suggest?params_b=8&goal=forgetting&minutes=10", headers=AUTH).json()
    assert b["context"]["goal"] == "forgetting" and any(r["name"] == "gsm8k" for r in b["recommendations"])
    for bad in ("params_b=-1", "params_b=abc", "goal=nope", "minutes=0", "data=weird"):
        path = "/api/methods/suggest" if bad.startswith(("params_b", "data")) else "/api/bench/suggest"
        assert c.get(f"{path}?{bad}", headers=AUTH).status_code == 422
    if server._manager is not None:
        server._manager.stop()
    server._manager = None


def test_ui_contains_the_new_panels_and_their_data_hooks():
    from pathlib import Path

    html = (Path(server.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    for needle in (
        "c_mem",
        "c_kv",
        "c_req",
        "c_tps",
        "/api/telemetry",
        "/api/methods/suggest",
        "/api/bench/suggest",
        "renderComps",
        "allocated up front",
    ):
        assert needle in html, needle


def test_requests_get_kv_figures_even_if_the_dashboard_was_never_opened(monkeypatch, tmp_path):
    class WithInfo(Fake):
        name = "with-info"

        def telemetry_info(self):
            return {
                "name": self.name,
                "kind": "llamacpp",
                "kv": {"bytes_per_token": 1024.0},
                "ctx_alloc_tokens": 100,
                "max_concurrency": 1,
            }

    monkeypatch.setenv("TINYFORGE_API_TOKEN", TOKEN)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "_backend", WithInfo())
    server.telemetry.requests.clear()
    server.telemetry.engine = {}
    c = TestClient(server.app)
    body = {"messages": [{"role": "user", "content": "a b c"}]}
    assert (
        c.post("/v1/chat/completions", json=body, headers=AUTH).status_code == 200
    )  # no /api/telemetry call first
    last = list(server.telemetry.requests)[-1]
    assert last["kv_used_mb"] is not None and server.telemetry.engine["kind"] == "llamacpp"
    if server._manager is not None:
        server._manager.stop()
    server._manager = None
