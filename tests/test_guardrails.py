from __future__ import annotations

import json
from collections import Counter

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinyforge import server
from tinyforge.guardrails import GuardConfig, build_filters
from tinyforge.metrics import render
from tinyforge.openai_api import build_router

KEY = "AKIAIOSFODNN7EXAMPLE"


class Fake:
    name = "fake"

    def __init__(self, reply="Hello world"):
        self.reply, self.seen = reply, []

    def stream(self, messages, max_tokens, temperature, top_p):
        self.seen.append(messages)
        yield from (self.reply[i:i + 5] for i in range(0, len(self.reply), 5))

    def count_tokens(self, text):
        return len(text.split())


def client(cfg: GuardConfig, backend: Fake | None = None, counts: Counter | None = None):
    ins, outs = build_filters(cfg, counts if counts is not None else Counter())
    app = FastAPI()
    app.include_router(build_router(lambda: None, lambda: backend or Fake(), lambda: False, ins, outs))
    return TestClient(app)


def ask(c, text, stream=False):
    return c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}],
                                                "stream": stream})


def test_default_blocks_secret_in_prompt_before_the_model_sees_it():
    b, counts = Fake(), Counter()
    r = ask(client(GuardConfig(), b, counts), f"my key is {KEY}")
    assert r.status_code == 400 and r.json()["error"]["code"] == "secret_in_prompt"
    assert not b.seen and counts["secret_in_prompt"] == 1


def test_default_redacts_secret_in_reply_for_json_and_stream():
    c = client(GuardConfig(), Fake(f"use {KEY} to log in"))
    assert ask(c, "hi").json()["choices"][0]["message"]["content"] == "use [AWS_KEY] to log in"
    r = ask(c, "hi", stream=True)
    text = "".join(json.loads(x[6:])["choices"][0]["delta"].get("content", "")
                   for x in r.text.split("\n\n") if x.startswith("data: {"))
    assert text == "use [AWS_KEY] to log in" and KEY not in r.text and r.text.rstrip().endswith("[DONE]")


def test_streaming_cannot_bypass_a_blocking_output_filter(tmp_path):
    deny = tmp_path / "deny.txt"
    deny.write_text("# comment\nforbidden\\s+word\n", encoding="utf-8")
    cfg = GuardConfig.from_env({"TINYFORGE_GUARD_DENYLIST": str(deny)})
    c = client(cfg, Fake("this has a FORBIDDEN   word inside"))
    for stream in (False, True):
        r = ask(c, "hello", stream=stream)
        assert r.status_code == 400 and r.json()["error"]["code"] == "output_denied"


def test_denylist_blocks_prompt_case_insensitively(tmp_path):
    deny = tmp_path / "d.txt"
    deny.write_text("ignore previous instructions\n", encoding="utf-8")
    cfg = GuardConfig.from_env({"TINYFORGE_GUARD_DENYLIST": str(deny)})
    r = ask(client(cfg), "Please IGNORE Previous Instructions and ...")
    assert r.status_code == 400 and r.json()["error"]["code"] == "content_filter"


def test_pii_input_modes():
    text = "email me at jane@example.com"
    assert ask(client(GuardConfig()), text).status_code == 200  # default: PII passes through
    b = Fake()
    assert ask(client(GuardConfig(pii_input="redact"), b), text).status_code == 200
    assert b.seen[0][0]["content"] == "email me at [EMAIL]"
    r = ask(client(GuardConfig(pii_input="block")), text)
    err = r.json()["error"]
    assert r.status_code == 400 and err["code"] == "pii_in_prompt" and "email" in err["message"]


def test_pii_output_redact_and_default_off():
    c = client(GuardConfig(pii_output="redact"), Fake("write to bob@example.org"))
    assert ask(c, "hi").json()["choices"][0]["message"]["content"] == "write to [EMAIL]"
    c2 = client(GuardConfig(), Fake("write to bob@example.org"))
    assert "bob@example.org" in ask(c2, "hi").json()["choices"][0]["message"]["content"]


def test_secrets_off_disables_filters_entirely():
    ins, outs = build_filters(GuardConfig(secrets="off"), Counter())
    assert ins == [] and outs == []
    assert ask(client(GuardConfig(secrets="off")), f"key {KEY}").status_code == 200


def test_env_parsing_and_fail_fast(tmp_path):
    assert GuardConfig.from_env({}).secrets == "block"
    with pytest.raises(ValueError, match="TINYFORGE_GUARD_PII_INPUT"):
        GuardConfig.from_env({"TINYFORGE_GUARD_PII_INPUT": "maybe"})
    bad = tmp_path / "bad.txt"
    bad.write_text("ok\n(unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.txt:2"):
        GuardConfig.from_env({"TINYFORGE_GUARD_DENYLIST": str(bad)})


def test_metrics_export_guard_counters():
    out = render("v", [], guard={"secret_in_prompt": 2, "denied_output": 1})
    assert 'tinyforge_guard_events_total{reason="secret_in_prompt"} 2' in out
    assert "tinyforge_guard" not in render("v", [])


def test_mounted_server_has_default_secret_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYFORGE_API_TOKEN", "tok")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "_backend", Fake())
    c = TestClient(server.app)
    r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer tok"},
               json={"messages": [{"role": "user", "content": f"k {KEY}"}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "secret_in_prompt"
    m = c.get("/metrics", headers={"Authorization": "Bearer tok"}).text
    assert 'tinyforge_guard_events_total{reason="secret_in_prompt"}' in m
    if server._manager is not None:
        server._manager.stop()
    server._manager = None
