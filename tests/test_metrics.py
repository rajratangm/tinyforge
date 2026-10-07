"""Prometheus /metrics: exposition-format validity, auth, and that the values reflect the job store."""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from tinyforge import server
from tinyforge.metrics import CONTENT_TYPE, STATUSES, latest_train_signals, render

TOKEN = "s3cret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# ---------------------------------------------------------------- a tiny exposition-format parser

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})? (\S+)$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _unescape(v: str) -> str:
    return re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), v)


def parse(text: str) -> tuple[dict[str, str], list[tuple[str, dict[str, str], float]]]:
    """Return ({family: type}, [(sample_name, labels, value)]); assert the structure rules of format 0.0.4."""
    assert text.endswith("\n") and not text.endswith("\n\n")
    helped: set[str] = set()
    types: dict[str, str] = {}
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        if line.startswith("# HELP "):
            helped.add(line.split(" ", 3)[2])
        elif line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ", 3)
            assert name in helped, f"TYPE before HELP for {name}"
            assert kind in ("gauge", "counter", "histogram", "summary", "untyped")
            assert name not in types, f"duplicate TYPE for {name}"
            types[name] = kind
        else:
            m = _SAMPLE.match(line)
            assert m, f"unparseable sample line: {line!r}"
            name, raw_labels, value = m.groups()
            labels = {k: _unescape(v) for k, v in _LABEL.findall(raw_labels or "")}
            if raw_labels:  # every byte of the label block must have been consumed by the label regex
                rebuilt = ",".join(f'{k}="{v}"' for k, v in _LABEL.findall(raw_labels))
                assert rebuilt == raw_labels, f"malformed label block: {raw_labels!r}"
            base = re.sub(r"_(bucket|sum|count)$", "", name) if name not in types else name
            assert name in types or base in types, f"sample {name} has no TYPE"
            samples.append((name, labels, float(value)))
    return types, samples


def value(samples, name, **labels) -> float:
    hits = [v for n, ls, v in samples if n == name and ls == labels]
    assert len(hits) == 1, f"{name}{labels}: {len(hits)} matches"
    return hits[0]


def names(samples) -> set[str]:
    return {n for n, _, _ in samples}


def job(status, started=None, finished=None, jid="aaaaaaaa") -> dict:
    return {"id": jid, "kind": "t", "status": status, "created": 0.0, "started": started,
            "finished": finished, "code": 0}


def step(n, loss=2.0, tok=900.0, mem=2.4):
    return json.dumps({"event": "step", "t": 1.0, "step": n, "loss": loss, "tok_per_s": tok,
                       "peak_mem_gb": mem})


# ---------------------------------------------------------------- rendering

def test_empty_store_is_valid_and_zeroed():
    types, samples = parse(render("1.2.3", []))
    assert types["tinyforge_jobs"] == "gauge" and types["tinyforge_job_duration_seconds"] == "histogram"
    assert value(samples, "tinyforge_build_info", version="1.2.3") == 1
    for s in STATUSES:
        assert value(samples, "tinyforge_jobs", status=s) == 0
    assert value(samples, "tinyforge_queue_depth") == 0
    assert value(samples, "tinyforge_job_duration_seconds_count", status="done") == 0
    assert not any(n.startswith(("tinyforge_train_", "tinyforge_val_")) for n in names(samples))


def test_label_values_are_escaped_and_round_trip():
    nasty = 'v"1\\2\nend'
    _, samples = parse(render(nasty, []))
    assert value(samples, "tinyforge_build_info", version=nasty) == 1


def test_status_counts_and_duration_histogram():
    jobs = [job("queued"), job("queued"), job("running", 10.0),
            job("done", 0.0, 5.0), job("done", 0.0, 100.0), job("failed", 0.0, 40.0),
            job("cancelled", None, 3.0), job("interrupted", 0.0, 9999.0)]
    _, samples = parse(render("v", jobs))
    assert [value(samples, "tinyforge_jobs", status=s) for s in STATUSES] == [2, 1, 2, 1, 1, 1]
    assert value(samples, "tinyforge_queue_depth") == 2

    b = lambda status, le: value(samples, "tinyforge_job_duration_seconds_bucket", status=status, le=le)  # noqa: E731
    assert (b("done", "10.0"), b("done", "60.0"), b("done", "300.0"), b("done", "+Inf")) == (1, 1, 2, 2)
    assert (b("failed", "30.0"), b("failed", "60.0"), b("failed", "+Inf")) == (0, 1, 1)
    assert value(samples, "tinyforge_job_duration_seconds_sum", status="done") == 105.0
    assert value(samples, "tinyforge_job_duration_seconds_count", status="failed") == 1
    # cancelled/interrupted must not leak into the duration histogram
    seen = {ls["status"] for n, ls, _ in samples if n.startswith("tinyforge_job_duration")}
    assert seen == {"done", "failed"}


def test_bucket_counts_are_cumulative():
    jobs = [job("done", 0.0, d) for d in (1, 20, 20, 500, 100000)]
    _, samples = parse(render("v", jobs))
    counts = [v for n, ls, v in samples
              if n == "tinyforge_job_duration_seconds_bucket" and ls["status"] == "done"]
    assert counts == sorted(counts) and counts[-1] == 5


# ---------------------------------------------------------------- training signals

def test_latest_step_and_eval_win_and_noise_is_ignored():
    log = ["starting up", step(5, loss=3.0), "not json {",
           json.dumps({"event": "eval", "step": 0, "val_loss": 9.0}),
           step(10, loss=2.5, tok=1000.0),
           json.dumps({"event": "eval", "step": 10, "val_loss": 2.2}),
           json.dumps({"no_event_key": 1}), "[1, 2, 3]"]
    sig = latest_train_signals(log)
    assert sig == {"step": 10, "loss": 2.5, "tok_per_second": 1000.0, "peak_mem_gb": 2.4, "val_loss": 2.2}


def test_nan_loss_is_rendered_and_bool_or_string_values_are_dropped():
    bad = json.dumps({"event": "step", "step": 3, "loss": float("nan"),
                      "tok_per_s": True, "peak_mem_gb": "2"})
    _, samples = parse(render("v", [job("running", 1.0)], [bad]))
    assert value(samples, "tinyforge_train_step") == 3
    assert value(samples, "tinyforge_train_loss") != value(samples, "tinyforge_train_loss")  # NaN
    assert not names(samples) & {"tinyforge_train_tok_per_second", "tinyforge_train_peak_mem_gb"}


def test_signals_omitted_once_training_has_finished_or_failed():
    for end in ("finished", "failed"):
        assert latest_train_signals([step(5), json.dumps({"event": end})]) == {}
    # a restart (new step after an old finished event) is current again
    assert latest_train_signals([json.dumps({"event": "finished"}), step(1)])["step"] == 1


def test_no_training_signals_without_a_running_job():
    _, samples = parse(render("v", [job("done", 0.0, 5.0)], [step(50)]))
    assert not any(n.startswith("tinyforge_train_") for n in names(samples))


# ---------------------------------------------------------------- HTTP: auth and live values

@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYFORGE_API_TOKEN", TOKEN)
    monkeypatch.delenv("TINYFORGE_AUTH", raising=False)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    yield tmp_path
    if server._manager is not None:
        server._manager.stop()
    server._manager = None


def test_metrics_requires_the_api_token(env, monkeypatch):
    c = TestClient(server.app)
    assert c.get("/metrics").status_code == 401
    assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = c.get("/metrics", headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"] == CONTENT_TYPE
    parse(r.text)

    monkeypatch.delenv("TINYFORGE_API_TOKEN")
    assert c.get("/metrics").status_code == 503  # secure by default: no token configured, no metrics
    monkeypatch.setenv("TINYFORGE_AUTH", "off")
    assert c.get("/metrics").status_code == 200  # explicit opt-in only


def test_metrics_reflect_store_and_only_running_job_signals(env):
    mgr = server.configure(env / "jobs.db", lambda *a: 0, before_job=lambda: None, autostart=False)
    c = TestClient(server.app)
    store = mgr.store

    old = store.enqueue("t", [["x"]])
    store.claim_next()
    store.finish(old, "done", 0)
    mgr.logs[old] = [step(999, loss=0.1)]  # finished jobs keep their log in memory; it must not be reported
    run_id = store.enqueue("t", [["x"]])
    store.enqueue("t", [["x"]])  # stays queued behind the running one
    store.claim_next()  # FIFO: the first enqueued job starts
    assert store.get(run_id)["status"] == "running"

    _, samples = parse(c.get("/metrics", headers=AUTH).text)
    assert value(samples, "tinyforge_jobs", status="done") == 1
    assert value(samples, "tinyforge_jobs", status="running") == 1
    assert value(samples, "tinyforge_jobs", status="queued") == 1
    assert value(samples, "tinyforge_queue_depth") == 1
    assert not any(n.startswith("tinyforge_train_") for n in names(samples))  # no live log yet

    mgr.logs[run_id] = ["boot", step(40, loss=1.5, tok=1234.0),
                        json.dumps({"event": "eval", "step": 40, "val_loss": 1.4})]
    _, samples = parse(c.get("/metrics", headers=AUTH).text)
    assert value(samples, "tinyforge_train_step") == 40
    assert value(samples, "tinyforge_train_loss") == 1.5
    assert value(samples, "tinyforge_train_tok_per_second") == 1234.0
    assert value(samples, "tinyforge_val_loss") == 1.4

    store.finish(run_id, "done", 0)  # job ends -> signals disappear even though its log is still in memory
    _, samples = parse(c.get("/metrics", headers=AUTH).text)
    assert not any(n.startswith(("tinyforge_train_", "tinyforge_val_")) for n in names(samples))
    assert value(samples, "tinyforge_jobs", status="running") == 0


def test_host_memory_gauges_rendered_only_when_given():
    assert "tinyforge_host" not in render("v", [])
    host = {"ram_total": 16_000_000_000, "ram_available": 4_000_000_000,
            "swap_used": 1_000_000_000, "junk": "x"}
    out = render("v", [], host=host)
    assert "tinyforge_host_ram_available_bytes 4000000000" in out
    assert "tinyforge_host_swap_used_bytes 1000000000" in out
