"""Serving engines: proxy, llama.cpp lifecycle, concurrency, errors. Uses a fake llama-server (no GPU)."""
from __future__ import annotations

import json
import sys
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinyforge.engines import GB, LlamaCppBackend, ProxyBackend, UpstreamError, choose_ngl, make_backend
from tinyforge.openai_api import build_router

FAKE = r'''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1]
open(os.environ["FAKE_ARGS"], "w").write(json.dumps(args))
key, port = opt("--api-key"), int(opt("--port"))
delay = float(os.environ.get("FAKE_DELAY", "0"))
mode = os.environ.get("FAKE_MODE", "ok")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _ok(self):
        return self.headers.get("Authorization") == "Bearer " + key
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path != "/health" and not self._ok():
            self.send_response(401); self.end_headers(); return
        if self.path == "/tokenize":
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"tokens":[1,2,3,4]}'); return
        if mode == "http500":
            self.send_response(500); self.end_headers(); self.wfile.write(b"engine exploded"); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for piece in ("Hel", "lo"):
            time.sleep(delay)
            chunk = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode()); self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_llama.py"
    script.write_text(FAKE, encoding="utf-8")
    args_file = tmp_path / "args.json"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    model = tmp_path / "m.gguf"
    model.write_bytes(b"0" * 1024)
    lora = tmp_path / "a.gguf"
    lora.write_bytes(b"0")
    made: list[LlamaCppBackend] = []

    def make(**kw):
        b = LlamaCppBackend(kw.pop("model", model), kw.pop("lora", lora),
                            server_bin=[sys.executable, str(script)], ngl=kw.pop("ngl", 7),
                            startup_timeout=30, **kw)
        made.append(b)
        return b

    yield make, args_file, model
    for b in made:
        b.stop()


def test_lazy_start_args_security_and_streaming(fake):
    make, args_file, _ = fake
    b = make(parallel=3)
    assert b.proc is None and not args_file.exists()  # nothing spawned until first use
    assert "".join(b.stream([{"role": "user", "content": "hi"}], 8, 0.0, 1.0)) == "Hello"
    args = json.loads(args_file.read_text())
    assert args[args.index("--host") + 1] == "127.0.0.1" and "--no-webui" in args
    assert args[args.index("-np") + 1] == "3" and args[args.index("-ngl") + 1] == "7" and "--lora" in args
    assert len(args[args.index("--api-key") + 1]) >= 24 and b.max_concurrency == 3
    assert b.name == "m+lora" and b.count_tokens("one two three") == 4


def test_upstream_requires_the_api_key(fake):
    make, _, _ = fake
    b = make()
    list(b.stream([{"role": "user", "content": "x"}], 4, 0.0, 1.0))  # start it
    wrong = ProxyBackend(b.base_url, api_key="wrong")
    with pytest.raises(UpstreamError, match="401"):
        list(wrong.stream([{"role": "user", "content": "x"}], 4, 0.0, 1.0))


def test_stop_and_restart_after_crash(fake):
    make, _, _ = fake
    b = make()
    list(b.stream([{"role": "user", "content": "x"}], 4, 0.0, 1.0))
    first = b.proc
    b.stop()
    assert first.poll() is not None and b.proc is None
    assert "".join(b.stream([{"role": "user", "content": "x"}], 4, 0.0, 1.0)) == "Hello"  # restarted
    b.proc.kill()
    b.proc.wait()
    assert "".join(b.stream([{"role": "user", "content": "x"}], 4, 0.0, 1.0)) == "Hello"


def test_missing_files_and_startup_failure(fake, tmp_path):
    make, _, _ = fake
    with pytest.raises(FileNotFoundError, match="GGUF model"):
        list(make(model=tmp_path / "nope.gguf").stream([{"role": "user", "content": "x"}], 4, 0.0, 1.0))
    with pytest.raises(FileNotFoundError, match="LoRA GGUF"):
        list(make(lora=tmp_path / "nope-lora.gguf").stream([{"role": "user", "content": "x"}], 4, 0.0, 1.0))
    crash = [sys.executable, "-c", "import sys; sys.exit(3)"]
    bad = LlamaCppBackend(tmp_path / "m.gguf", None, server_bin=crash, ngl=0, startup_timeout=10)
    with pytest.raises(UpstreamError, match="exited during startup"):
        list(bad.stream([{"role": "user", "content": "x"}], 4, 0.0, 1.0))


def test_choose_ngl():
    assert choose_ngl(2 * GB, 0) == 0                      # no GPU info -> CPU
    assert choose_ngl(2 * GB, 8 * GB) == 999               # fits with headroom -> everything
    assert choose_ngl(int(4.9 * GB), int(3.3 * GB)) == 18  # the measured 8B / 4 GB case -> a safe share
    assert choose_ngl(8 * GB, int(0.4 * GB)) == 0          # nothing fits -> CPU


def test_router_uses_engine_concurrency_and_maps_errors(fake, monkeypatch):
    make, _, _ = fake
    monkeypatch.setenv("FAKE_DELAY", "0.4")
    b = make(parallel=2)
    b.queue_wait_s = 0.0  # reject at once when both slots are busy
    app = FastAPI()
    app.include_router(build_router(lambda: None, lambda: b))
    c = TestClient(app)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    codes: list[int] = []
    ts = [threading.Thread(target=lambda: codes.append(c.post("/v1/chat/completions", json=body).status_code))
          for _ in range(2)]
    [t.start() for t in ts]
    time.sleep(0.25)
    assert c.post("/v1/chat/completions", json=body).status_code == 429  # 3rd request: both slots busy
    [t.join(10) for t in ts]
    assert codes == [200, 200]
    assert c.post("/v1/chat/completions", json=body).status_code == 200  # slots released


def test_waiting_request_gets_a_slot_when_one_frees_up_within_queue_wait(fake, monkeypatch):
    make, _, _ = fake
    monkeypatch.setenv("FAKE_DELAY", "0.4")
    b = make(parallel=1)
    b.queue_wait_s = 10.0
    app = FastAPI()
    app.include_router(build_router(lambda: None, lambda: b))
    c = TestClient(app)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    codes: list[int] = []
    ts = [threading.Thread(target=lambda: codes.append(c.post("/v1/chat/completions", json=body).status_code))
          for _ in range(3)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert codes == [200, 200, 200]  # one slot, three requests: they queue instead of failing


def test_router_maps_upstream_and_missing_model_errors(fake, monkeypatch, tmp_path):
    make, _, _ = fake
    monkeypatch.setenv("FAKE_MODE", "http500")
    b = make()
    app = FastAPI()
    app.include_router(build_router(lambda: None, lambda: b))
    c = TestClient(app)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"
    s = c.post("/v1/chat/completions", json={**body, "stream": True})
    assert s.status_code == 200 and "upstream_error" in s.text and s.text.rstrip().endswith("[DONE]")
    gone = make(model=tmp_path / "gone.gguf")
    app2 = FastAPI()
    app2.include_router(build_router(lambda: None, lambda: gone))
    r2 = TestClient(app2).post("/v1/chat/completions", json=body)
    assert r2.status_code == 404 and r2.json()["error"]["code"] == "model_not_found"


def test_make_backend_from_env(tmp_path):
    sentinel = object()
    assert make_backend({}, sentinel) is sentinel
    assert make_backend({"TINYFORGE_ENGINE": "hf"}, sentinel) is sentinel
    with pytest.raises(ValueError, match="TINYFORGE_GGUF"):
        make_backend({"TINYFORGE_ENGINE": "llamacpp"})
    with pytest.raises(ValueError, match="hf or llamacpp"):
        make_backend({"TINYFORGE_ENGINE": "vllm-someday"})
    b = make_backend({"TINYFORGE_ENGINE": "llamacpp", "TINYFORGE_GGUF": str(tmp_path / "x.gguf"),
                      "TINYFORGE_NGL": "12", "TINYFORGE_PARALLEL": "4"})
    assert isinstance(b, LlamaCppBackend) and b.ngl == 12 and b.max_concurrency == 4 and b.proc is None
    b.stop()
