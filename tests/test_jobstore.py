"""Durable queue semantics: FIFO, single running job, restart recovery."""

from __future__ import annotations

import pytest

from tinyforge.jobstore import JobStore


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


def test_fifo_and_single_running(store):
    a, b = store.enqueue("t", [["x"]]), store.enqueue("t", [["y"]])
    first = store.claim_next()
    assert first["id"] == a and first["status"] == "running" and first["stages"] == [["x"]]
    assert store.claim_next() is None  # one at a time: b waits while a runs
    store.finish(a, "done", 0, ["l1", "l2"])
    assert store.get(a)["status"] == "done" and store.get(a)["log_tail"] == "l1\nl2"
    assert store.claim_next()["id"] == b


def test_queued_survives_restart_running_becomes_interrupted(tmp_path):
    db = tmp_path / "jobs.db"
    s1 = JobStore(db)
    running, queued = s1.enqueue("t", [["a"]]), s1.enqueue("t", [["b"]])
    assert s1.claim_next()["id"] == running
    s1.log_path(running).write_text("line one\nline two\n", encoding="utf-8")

    s2 = JobStore(db)  # simulated restart: a new process opening the same files
    assert s2.recover() == 1
    r = s2.get(running)
    assert r["status"] == "interrupted" and r["finished"] is not None
    assert r["log_tail"] == "line one\nline two"
    assert s2.get(queued)["status"] == "queued"
    assert s2.claim_next()["id"] == queued  # resumed; the interrupted job is NOT re-run
    assert s2.recover() == 1  # and the one we just claimed is 'running' again, as expected
    assert s2.get(running)["status"] == "interrupted"


def test_cancel_only_affects_queued(store):
    a, b = store.enqueue("t", [["x"]]), store.enqueue("t", [["y"]])
    store.claim_next()
    assert store.cancel_queued(b) is True
    assert store.cancel_queued(a) is False  # already running
    assert store.cancel_queued("nonexist") is False
    assert store.get(b)["status"] == "cancelled"


def test_log_path_rejects_traversal(store):
    for bad in ("../etc/passwd", "ABCDEFGH", "short", "x" * 8 + "/.."):
        with pytest.raises(ValueError):
            store.log_path(bad)


def test_list_order_and_has_running(store):
    ids = [store.enqueue("t", [["x"]]) for _ in range(3)]
    assert [j["id"] for j in store.list()] == ids
    assert store.has_running() is False
    store.claim_next()
    assert store.has_running() is True
