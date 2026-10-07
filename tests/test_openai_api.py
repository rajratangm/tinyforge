from __future__ import annotations

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinyforge import server
from tinyforge.openai_api import Blocked, apply_stop, build_router

TOKEN = "t0k3n-for-tests"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Fake:
    name = "fake-model"

    def __init__(self, pieces=("Hel", "lo ", "wor", "ld")):
        self.pieces = pieces
        self.seen: list = []

    def stream(self, messages, max_tokens, temperature, top_p):
        self.seen.append((messages, max_tokens, temperature, top_p))
        yield from self.pieces

    def count_tokens(self, text):
        return len(text.split())


def _client(backend=None, busy=lambda: False, in_f=None, out_f=None):
    app = FastAPI()
    app.include_router(build_router(lambda: None, lambda: backend or Fake(), busy, in_f, out_f))
    return TestClient(app)


BODY = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}


def test_models_and_basic_completion_shape():
    c = _client()
    assert c.get("/v1/models").json()["data"][0]["id"] == "fake-model"
    r = c.post("/v1/chat/completions", json=BODY).json()
    assert r["object"] == "chat.completion" and r["choices"][0]["message"] == {
        "role": "assistant", "content": "Hello world"}
    assert r["choices"][0]["finish_reason"] == "stop" and r["usage"]["total_tokens"] >= 1


def test_streaming_sse_chunks_then_done():
    r = _client().post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [x[6:] for x in r.text.split("\n\n") if x.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(x) for x in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_length_finish_reason_when_hitting_max_tokens():
    b = Fake(pieces=("a b c d e f",))
    r = _client(b).post("/v1/chat/completions", json={**BODY, "max_tokens": 6}).json()
    assert r["choices"][0]["finish_reason"] == "length"


def test_stop_sequences_split_across_deltas():
    cases = ((["ab", "c", "STOPxyz"], "abc"), (["abST", "OPzz"], "ab"), (["no stop here"], "no stop here"))
    for pieces, want in cases:
        out = list(apply_stop(iter(pieces), ["STOP"]))
        assert "".join(t for t, _ in out) == want and any(s for _, s in out) == (pieces != ["no stop here"])
    r = _client().post("/v1/chat/completions", json={**BODY, "stop": "wor"}).json()
    assert r["choices"][0]["message"]["content"] == "Hello " and r["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("patch", [
    {"n": 2}, {"max_tokens": 99999}, {"temperature": 5}, {"messages": []},
    {"messages": [{"role": "assistant", "content": "x"}]},
    {"messages": [{"role": "tool", "content": "x"}]},
    {"messages": [{"role": "user", "content": "x" * 40_000}]},
])
def test_bad_requests_are_rejected_without_calling_the_backend(patch):
    b = Fake()
    r = _client(b).post("/v1/chat/completions", json={**BODY, **patch})
    assert r.status_code in (400, 422) and not b.seen


def test_gpu_busy_and_missing_model_are_clear_errors():
    r = _client(busy=lambda: True).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 503 and r.json()["error"]["code"] == "gpu_busy" and "Retry-After" in r.headers

    def missing():
        raise FileNotFoundError("No fine-tuned model yet")

    app = FastAPI()
    app.include_router(build_router(lambda: None, missing))
    r = TestClient(app).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_second_concurrent_generation_gets_429_and_lock_is_released():
    gate, started = threading.Event(), threading.Event()

    class Slow(Fake):
        def stream(self, *a):
            started.set()
            gate.wait(5)
            yield "x"

    c = _client(Slow())
    out: list = []
    t = threading.Thread(target=lambda: out.append(c.post("/v1/chat/completions", json=BODY)))
    t.start()
    assert started.wait(5)
    r2 = c.post("/v1/chat/completions", json=BODY)
    assert r2.status_code == 429 and r2.headers["Retry-After"]
    gate.set()
    t.join(5)
    assert out[0].status_code == 200
    assert _client(Fake()).post("/v1/chat/completions", json=BODY).status_code == 200


def test_guardrail_hooks_block_input_and_rewrite_output():
    def deny(msgs):
        if "secret" in msgs[-1]["content"]:
            raise Blocked("nope")

    c = _client(in_f=[deny], out_f=[lambda t: t.replace("world", "[redacted]")])
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "the secret"}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "content_filter"
    ok = c.post("/v1/chat/completions", json=BODY).json()
    assert ok["choices"][0]["message"]["content"] == "Hello [redacted]"


def test_mounted_in_server_requires_auth_and_serves(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYFORGE_API_TOKEN", TOKEN)
    monkeypatch.delenv("TINYFORGE_AUTH", raising=False)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "_backend", Fake())
    c = TestClient(server.app)
    assert c.get("/v1/models").status_code == 401
    assert c.post("/v1/chat/completions", json=BODY).status_code == 401
    r = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Hello world"
    if server._manager is not None:
        server._manager.stop()
    server._manager = None
