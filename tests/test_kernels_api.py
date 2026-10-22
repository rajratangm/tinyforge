"""The Kernels panel's API: status, the latest run's kernel report, recorded benchmarks, on-demand verify."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tinyforge import server
from tinyforge.kernels import bench, env

TOKEN = "s3cret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYFORGE_API_TOKEN", TOKEN)
    monkeypatch.delenv("TINYFORGE_AUTH", raising=False)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(env, "triton_status",
                        lambda: env.KernelEnv(True, "", "Test GPU", "8.6", "3.2.0"))
    server._kernel_env_cache.clear()
    yield TestClient(server.app), tmp_path
    if server._manager is not None:
        server._manager.stop()
    server._manager = None


def test_kernels_endpoint_requires_the_api_token(client):
    c, _ = client
    assert c.get("/api/kernels").status_code == 401
    assert c.post("/api/kernels/verify").status_code == 401


def test_status_lists_triton_and_the_builtin_kernels(client):
    c, _ = client
    r = c.get("/api/kernels", headers=AUTH)
    assert r.status_code == 200
    d = r.json()
    assert d["triton"] == {"ok": True, "reason": "", "gpu": "Test GPU", "compute_capability": "8.6",
                           "triton_version": "3.2.0"}
    names = [k["name"] for k in d["kernels"]]
    assert names == ["rmsnorm", "swiglu", "rope", "fused_ce", "add_rmsnorm"]
    assert all(k["what"] for k in d["kernels"])
    assert d["report"] is None and d["benchmarks"] == []


def test_status_is_cached_so_the_page_does_not_recompile_a_kernel_every_refresh(client, monkeypatch):
    c, _ = client
    calls = []
    monkeypatch.setattr(env, "triton_status", lambda: calls.append(1) or env.KernelEnv(True))
    server._kernel_env_cache.clear()
    c.get("/api/kernels", headers=AUTH)
    c.get("/api/kernels", headers=AUTH)
    assert len(calls) == 1


def test_latest_run_report_and_benchmarks_are_returned(client):
    c, root = client
    (root / "runs" / "ft").mkdir(parents=True)
    report = {"patched": ["rmsnorm", "swiglu"], "skipped": {"rope": "not adopted"},
              "speedups": {"rmsnorm": 2.5}}
    (root / "runs" / "ft" / "kernel_report.json").write_text(json.dumps(report))
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "kernels-phase1-demo.json").write_text(json.dumps({
        "model": "org/demo", "steps": 30, "off": {"tok_per_s": 1000.0, "peak_mem_gb": 2.0},
        "auto": {"tok_per_s": 1150.0, "peak_mem_gb": 1.5}, "speedup": 1.15, "vram_saved_gb": 0.5,
        "loss_close": True}))
    (root / "benchmarks" / "kernels-broken.json").write_text("{not json")
    d = c.get("/api/kernels", headers=AUTH).json()
    assert d["report"] == report
    assert d["benchmarks"] == [{"file": "kernels-phase1-demo.json", "model": "org/demo", "steps": 30,
                                "speedup": 1.15, "vram_saved_gb": 0.5, "loss_close": True,
                                "off_peak_gb": 2.0, "auto_peak_gb": 1.5}]


def test_verify_runs_the_checker_and_returns_a_table(client, monkeypatch):
    c, _ = client
    rows = [{"kernel": "rmsnorm", "shape": [2048, 960], "dtype": "bfloat16", "ok": True, "max_err": 0.01,
             "grad_err": 0.5, "reason": ""}]
    monkeypatch.setattr(bench, "verify_rows", lambda *a, **k: rows)
    d = c.post("/api/kernels/verify", headers=AUTH).json()
    assert d["ok"] is True and d["rows"] == rows
    assert "shapes and dtype" in d["note"]


def test_verify_reports_failure_and_refuses_while_a_job_runs(client, monkeypatch):
    c, _ = client
    bad = [{"kernel": "rope", "shape": [1], "dtype": "bfloat16", "ok": False, "reason": "gradient differs"}]
    monkeypatch.setattr(bench, "verify_rows", lambda *a, **k: bad)
    assert c.post("/api/kernels/verify", headers=AUTH).json()["ok"] is False
    monkeypatch.setattr(server, "_busy", lambda: True)
    assert c.post("/api/kernels/verify", headers=AUTH).status_code == 409


def test_verify_without_a_gpu_is_a_clear_error_not_a_crash(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(env, "triton_status", lambda: env.KernelEnv(False, "CUDA is not available"))
    server._kernel_env_cache.clear()
    r = c.post("/api/kernels/verify", headers=AUTH)
    assert r.status_code == 400 and "CUDA is not available" in r.json()["detail"]


def test_the_page_has_a_kernels_card():
    html = (server.Path(server.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="kernels"' in html and "/api/kernels" in html
