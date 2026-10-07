"""SQL execution accuracy against an OpenAI-compatible server (llama-server, our /v1 API, vLLM later).

Gold and predicted SQL are both run on the REAL table and their result sets compared (unordered unless the
gold SQL has ORDER BY). Variants: `tuned` (LoRA scale 1), `base` (scale 0, raw prompt) and `base_instructed`
(scale 0 plus an explicit "reply with one SQLite query" instruction: the fair baseline, because a base chat
model otherwise answers in prose). Adapter scaling uses llama-server's `/lora-adapters` endpoint and is
skipped for other servers.

Limits: one run, greedy decoding, no confidence interval; a model that learned the question templates can
score near 100% here without being a general SQL writer, so also score shapes it never trained on
(`data tabular --hard`).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from .tabular import load_csv

INSTRUCTION = "Reply with exactly one SQLite query and nothing else (no explanation, no markdown).\n\n"
VARIANTS = ("tuned", "base", "base_instructed")


def clean_sql(sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip(), flags=re.I)
    return sql.strip().rstrip(";").strip()


def _rows(db: sqlite3.Connection, sql: str, ordered: bool) -> list:
    r = [tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in db.execute(sql).fetchall()]
    return r if ordered else sorted(r, key=repr)


def score(db: sqlite3.Connection, test: list[dict], preds: list[str]) -> tuple[dict, list[dict]]:
    ok = err = exact = 0
    misses: list[dict] = []
    for ex, raw in zip(test, preds, strict=True):
        q, gold = ex["messages"][0]["content"], ex["messages"][1]["content"]
        pred = clean_sql(raw)
        exact += pred.lower().split() == gold.lower().split()
        ordered = "ORDER BY" in gold.upper()
        try:
            good = _rows(db, pred, ordered) == _rows(db, gold, ordered)
        except sqlite3.Error:
            err += 1
            misses.append({"q": q.split("\n")[0], "gold": gold, "pred": pred[:200], "why": "error"})
            continue
        ok += good
        if not good and len(misses) < 8:
            misses.append({"q": q.split("\n")[0], "gold": gold, "pred": pred[:200], "why": "wrong result"})
    n = max(1, len(test))
    return {"exec_accuracy": ok / n, "sql_error_rate": err / n, "exact_match": exact / n}, misses[:6]


def _ask_http(url: str, prompt: str, timeout: float = 600.0) -> str:
    r = httpx.post(
        f"{url}/v1/chat/completions",
        timeout=timeout,
        json={"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 96},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _set_lora_scale(url: str, scale: float) -> None:
    r = httpx.post(f"{url}/lora-adapters", json=[{"id": 0, "scale": scale}], timeout=60)
    r.raise_for_status()


def run(
    server_url: str,
    csv_path: Path,
    test_path: Path,
    n: int = 100,
    variants: tuple[str, ...] = ("tuned", "base_instructed"),
    ask: Callable[[str, str], str] = _ask_http,
    set_scale: Callable[[str, float], None] = _set_lora_scale,
) -> dict:
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise ValueError(f"unknown variant(s) {bad}; choose from {VARIANTS}")
    t = load_csv(csv_path)
    db = sqlite3.connect(":memory:")
    db.execute(t.create_sql())
    db.executemany(f"INSERT INTO {t.name} VALUES ({','.join('?' * len(t.columns))})", t.rows)
    test = [json.loads(x) for x in test_path.read_text(encoding="utf-8").splitlines() if x.strip()][:n]
    url = server_url.rstrip("/")
    res: dict = {"n": len(test), "table": t.name, "table_rows": len(t.rows), "server": url}
    for v in variants:
        try:
            set_scale(url, 1.0 if v == "tuned" else 0.0)
        except httpx.HTTPError:
            if v != "tuned":
                raise
        t0 = time.time()
        preds = [
            ask(url, (INSTRUCTION if v == "base_instructed" else "") + ex["messages"][0]["content"])
            for ex in test
        ]
        metrics, misses = score(db, test, preds)
        res[v] = {**metrics, "seconds": round(time.time() - t0, 1), "sample_misses": misses}
    return res
