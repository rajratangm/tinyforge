"""API auth (secure by default) and the durable job queue, via the real HTTP app with a fake stage runner."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from tinyforge import server
from tinyforge.jobstore import JobStore

TOKEN = "s3cret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYFORGE_API_TOKEN", TOKEN)
    monkeypatch.delenv("TINYFORGE_AUTH", raising=False)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    yield tmp_path
    if server._manager is not None:
        server._manager.stop()
    server._manager = None


def fake_runner(calls: list, fail_on: str | None = None):
    def run(job_id, argv, emit):
        calls.append(argv[0])
        emit(f"hello {argv[0]}")
        return 1 if argv[0] == fail_on else 0

    return run


def wait_status(c, job_id, want, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        jobs = {j["id"]: j for j in c.get("/api/jobs", headers=AUTH).json()}
        if jobs[job_id]["status"] in want:
            return jobs[job_id]
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {want}: {jobs[job_id]}")


# ---------------------------------------------------------------- auth


def test_refuses_when_no_token_configured(monkeypatch, env):
    monkeypatch.delenv("TINYFORGE_API_TOKEN")
    c = TestClient(server.app)
    r = c.get("/api/runs")
    assert r.status_code == 503 and "TINYFORGE_API_TOKEN" in r.json()["detail"]
    assert c.get("/healthz").status_code == 200


def test_auth_off_is_explicit_opt_in(monkeypatch, env):
    monkeypatch.delenv("TINYFORGE_API_TOKEN")
    monkeypatch.setenv("TINYFORGE_AUTH", "off")
    assert TestClient(server.app).get("/api/runs").status_code == 200


def test_token_required_and_checked(env):
    c = TestClient(server.app)
    assert c.get("/api/runs").status_code == 401
    r = c.get("/api/runs", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert c.get("/api/runs", headers={"Authorization": TOKEN}).status_code == 401  # no scheme
    assert c.get("/api/runs", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert c.get("/api/runs", headers={"Authorization": f"Bearer {TOKEN}x"}).status_code == 401
    assert c.get("/api/runs", headers=AUTH).status_code == 200


def test_every_api_route_is_protected_but_health_and_ui_are_open(env):
    c = TestClient(server.app)
    # OpenAPI lists every route however it was registered, so a route added straight to `app` is caught too.
    paths = {p: ms for p, ms in server.app.openapi()["paths"].items() if p.startswith("/api/")}
    assert len(paths) >= 8
    for path, methods in paths.items():
        for method in methods:
            url = path.replace("{job_id}", "abcd1234")
            status = c.request(method.upper(), url).status_code
            assert status == 401, f"{method.upper()} {path} is unauthenticated (got {status})"
    assert c.get("/healthz").status_code == 200
    assert "tinyforge" in c.get("/").text


# ---------------------------------------------------------------- queue


def test_pipeline_job_runs_through_api_and_logs_persist(env):
    calls: list = []
    server.configure(env / "jobs.db", fake_runner(calls), before_job=lambda: None)
    c = TestClient(server.app)
    job_id = c.post("/api/jobs/pipeline", json={"steps": 5}, headers=AUTH).json()["id"]
    job = wait_status(c, job_id, ("done", "failed"))
    assert job["status"] == "done" and job["code"] == 0 and calls == ["data", "train", "eval"]

    body = c.get(f"/api/jobs/{job_id}/events", headers=AUTH).text
    assert 'data: "hello data"' in body and "event: end\ndata: done" in body

    server.configure(env / "jobs.db", fake_runner([]), before_job=lambda: None)  # restart: memory logs gone
    again = c.get(f"/api/jobs/{job_id}/events", headers=AUTH).text
    assert 'data: "hello train"' in again and "event: end\ndata: done" in again  # served from the log file


def test_failed_stage_stops_the_job(env):
    calls: list = []
    server.configure(env / "jobs.db", fake_runner(calls, fail_on="train"), before_job=lambda: None)
    c = TestClient(server.app)
    job_id = c.post("/api/jobs/pipeline", json={"steps": 5}, headers=AUTH).json()["id"]
    job = wait_status(c, job_id, ("done", "failed"))
    assert job["status"] == "failed" and job["code"] == 1 and calls == ["data", "train"]


def test_queued_jobs_survive_restart_and_run_in_order(env):
    db, calls = env / "jobs.db", []
    server.configure(db, fake_runner(calls), before_job=lambda: None, autostart=False)  # nothing runs yet
    c = TestClient(server.app)
    first = c.post("/api/jobs/pipeline", json={"steps": 1}, headers=AUTH).json()["id"]
    second = c.post("/api/jobs/pipeline", json={"steps": 2}, headers=AUTH).json()["id"]
    assert [j["status"] for j in c.get("/api/jobs", headers=AUTH).json()] == ["queued", "queued"]

    server.configure(db, fake_runner(calls), before_job=lambda: None)  # "restart" with autostart
    wait_status(c, second, ("done",))
    assert wait_status(c, first, ("done",))["finished"] <= wait_status(c, second, ("done",))["started"]
    assert calls == ["data", "train", "eval"] * 2


def test_running_job_is_interrupted_not_rerun_after_restart(env):
    db, calls = env / "jobs.db", []
    crashed = JobStore(db)
    stale = crashed.enqueue("pipeline", [["data"]])
    assert crashed.claim_next()["id"] == stale  # the "server" dies while this is running
    waiting = crashed.enqueue("pipeline", [["data"]])

    server.configure(db, fake_runner(calls), before_job=lambda: None)
    c = TestClient(server.app)
    assert wait_status(c, stale, ("interrupted",))["status"] == "interrupted"
    assert wait_status(c, waiting, ("done",))["status"] == "done"
    assert calls == ["data"]  # only the queued job ran; the interrupted one was not re-run


def test_cancel_queued_and_running(env):
    gate, started = threading.Event(), threading.Event()

    def blocking(job_id, argv, emit):
        started.set()
        gate.wait(10)
        return 0

    server.configure(env / "jobs.db", blocking, before_job=lambda: None)
    c = TestClient(server.app)
    running = c.post("/api/jobs/pipeline", json={"steps": 1}, headers=AUTH).json()["id"]
    assert started.wait(5)
    queued = c.post("/api/jobs/pipeline", json={"steps": 1}, headers=AUTH).json()["id"]
    assert wait_status(c, queued, ("queued",))["status"] == "queued"

    assert c.post(f"/api/jobs/{queued}/cancel", headers=AUTH).status_code == 200
    assert wait_status(c, queued, ("cancelled",))["status"] == "cancelled"
    assert c.post(f"/api/jobs/{running}/cancel", headers=AUTH).status_code == 200
    gate.set()
    assert wait_status(c, running, ("cancelled",))["status"] == "cancelled"
    assert c.post("/api/jobs/deadbeef/cancel", headers=AUTH).status_code == 404
    assert c.post("/api/jobs/not-an-id/cancel", headers=AUTH).status_code == 404


def test_unknown_job_events_404_and_busy_guard(env):
    server.configure(env / "jobs.db", fake_runner([]), before_job=lambda: None, autostart=False)
    c = TestClient(server.app)
    assert c.get("/api/jobs/deadbeef/events", headers=AUTH).status_code == 404
    assert c.get("/api/jobs/..%2F..%2Fetc/events", headers=AUTH).status_code in (404, 422)
    server._manager.store.enqueue("t", [["x"]])
    server._manager.store.claim_next()  # a job is running -> chat endpoints must refuse (single GPU)
    assert c.post("/api/generate", json={"run": "micro"}, headers=AUTH).status_code == 409
