"""Execution accuracy against an OpenAI-compatible server (llama-server with a LoRA adapter loaded).

usage: python eval_server.py http://127.0.0.1:8099 table.csv test.jsonl N out.json [variants]
variants (comma list): tuned (adapter scale 1), base (scale 0, raw prompt), base_instructed (scale 0 + SQL-only instruction)
Gold and predicted SQL are both run on the REAL table and their result sets compared.
"""
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import httpx

from tinyforge.tabular import load_csv

url, csv_path, test_path, n, out_json = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
variants = (sys.argv[6] if len(sys.argv) > 6 else "tuned,base_instructed").split(",")
INSTR = "Reply with exactly one SQLite query and nothing else (no explanation, no markdown).\n\n"

t = load_csv(csv_path)
db = sqlite3.connect(":memory:")
db.execute(t.create_sql())
db.executemany(f"INSERT INTO {t.name} VALUES ({','.join('?' * len(t.columns))})", t.rows)
test = [json.loads(x) for x in test_path.read_text(encoding="utf-8").splitlines() if x][:n]


def rows_of(sql, ordered):
    r = [tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in db.execute(sql).fetchall()]
    return r if ordered else sorted(r, key=repr)


def clean(sql):
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip(), flags=re.I)
    return sql.strip().rstrip(";").strip()


def set_scale(scale):
    r = httpx.post(f"{url}/lora-adapters", json=[{"id": 0, "scale": scale}], timeout=60)
    r.raise_for_status()


def ask(prompt):
    r = httpx.post(f"{url}/v1/chat/completions", timeout=600, json={
        "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 96})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


res = {"n": len(test), "table": t.name, "table_rows": len(t.rows), "server": url}
for v in variants:
    set_scale(1.0 if v == "tuned" else 0.0)
    t0, ok, err, exact, misses = time.time(), 0, 0, 0, []
    for ex in test:
        q, gold = ex["messages"][0]["content"], ex["messages"][1]["content"]
        pred = clean(ask((INSTR if v == "base_instructed" else "") + q))
        exact += pred.lower().split() == gold.lower().split()
        ordered = "ORDER BY" in gold.upper()
        try:
            good = rows_of(pred, ordered) == rows_of(gold, ordered)
        except sqlite3.Error:
            err += 1
            misses.append({"q": q.split("\n")[0], "gold": gold, "pred": pred[:200], "why": "error"})
            continue
        ok += good
        if not good and len(misses) < 8:
            misses.append({"q": q.split("\n")[0], "gold": gold, "pred": pred[:200], "why": "wrong result"})
    k = len(test)
    res[v] = {"exec_accuracy": ok / k, "sql_error_rate": err / k, "exact_match": exact / k,
              "seconds": round(time.time() - t0, 1), "sample_misses": misses[:6]}
    print(v, {a: b for a, b in res[v].items() if a != "sample_misses"}, flush=True)
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
