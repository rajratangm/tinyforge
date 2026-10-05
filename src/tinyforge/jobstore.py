"""Durable FIFO job queue on SQLite (stdlib, WAL). One writer process is assumed (the API server).

Invariants: at most one job is 'running' (single GPU); jobs are claimed oldest-first; a job found 'running'
at startup is marked 'interrupted' and is never silently re-run; 'queued' jobs survive a restart.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

TERMINAL = ("done", "failed", "cancelled", "interrupted")
_ID = re.compile(r"^[0-9a-f]{8}$")
LOG_TAIL_LINES = 40

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  seq      INTEGER PRIMARY KEY AUTOINCREMENT,
  id       TEXT NOT NULL UNIQUE,
  kind     TEXT NOT NULL,
  status   TEXT NOT NULL,
  stages   TEXT NOT NULL,
  created  REAL NOT NULL,
  started  REAL,
  finished REAL,
  code     INTEGER,
  log_tail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS jobs_status_seq ON jobs (status, seq);
"""


class JobStore:
    def __init__(self, db_path: Path, log_dir: Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.log_dir = Path(log_dir) if log_dir else self.db_path.parent / "_jobs"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # One-time setup runs outside _tx(): journal_mode and executescript manage transactions themselves.
        with closing(sqlite3.connect(self.db_path, timeout=10, isolation_level=None)) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        # A short-lived connection per operation: safe across threads, and each `with` is one transaction.
        with closing(sqlite3.connect(self.db_path, timeout=10, isolation_level=None)) as c:
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
            except BaseException:
                c.execute("ROLLBACK")
                raise
            else:
                c.execute("COMMIT")

    @staticmethod
    def _row(r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        d["stages"] = json.loads(d["stages"])
        return d

    def enqueue(self, kind: str, stages: list[list[str]]) -> str:
        job_id = uuid.uuid4().hex[:8]
        with self._tx() as c:
            c.execute("INSERT INTO jobs (id, kind, status, stages, created) VALUES (?, ?, 'queued', ?, ?)",
                      (job_id, kind, json.dumps(stages), time.time()))
        return job_id

    def claim_next(self) -> dict | None:
        """Atomically move the oldest queued job to running, unless one is already running."""
        with self._tx() as c:
            if c.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone():
                return None
            r = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY seq LIMIT 1").fetchone()
            if r is None:
                return None
            c.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), r["id"]))
            return self._row(c.execute("SELECT * FROM jobs WHERE id=?", (r["id"],)).fetchone())

    def finish(self, job_id: str, status: str, code: int | None, log_tail: list[str] | None = None) -> None:
        assert status in TERMINAL, status
        tail = "\n".join((log_tail or [])[-LOG_TAIL_LINES:])
        with self._tx() as c:
            c.execute("UPDATE jobs SET status=?, code=?, finished=?, log_tail=? WHERE id=?",
                      (status, code, time.time(), tail, job_id))

    def cancel_queued(self, job_id: str) -> bool:
        with self._tx() as c:
            cur = c.execute("UPDATE jobs SET status='cancelled', finished=? WHERE id=? AND status='queued'",
                            (time.time(), job_id))
            return cur.rowcount == 1

    def recover(self) -> int:
        """Call once at server start. Jobs left 'running' by a crash/restart become 'interrupted'."""
        with self._tx() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM jobs WHERE status='running'")]
            for job_id in ids:
                c.execute("UPDATE jobs SET status='interrupted', finished=?, log_tail=? WHERE id=?",
                          (time.time(), "\n".join(self.read_log(job_id)[-LOG_TAIL_LINES:]), job_id))
        return len(ids)

    def get(self, job_id: str) -> dict | None:
        with self._tx() as c:
            return self._row(c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, limit: int = 200) -> list[dict]:
        with self._tx() as c:
            rows = c.execute("SELECT * FROM jobs ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in reversed(rows)]  # type: ignore[misc]

    def has_running(self) -> bool:
        with self._tx() as c:
            return c.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone() is not None

    # ---- per-job log files (survive restarts)
    def log_path(self, job_id: str) -> Path:
        if not _ID.match(job_id):
            raise ValueError("bad job id")
        return self.log_dir / f"{job_id}.log"

    def read_log(self, job_id: str) -> list[str]:
        p = self.log_path(job_id)
        return p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []
