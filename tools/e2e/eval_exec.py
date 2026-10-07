"""Execution accuracy on the REAL table: run generated SQL and the gold SQL, compare result sets."""
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

from tinyforge.ft_eval import _gen
from tinyforge.openai_api import HFBackend
from tinyforge.tabular import load_csv

run_dir, csv_path, n, out_json = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
t = load_csv(csv_path)
db = sqlite3.connect(":memory:")
db.execute(t.create_sql())
db.executemany(f"INSERT INTO {t.name} VALUES ({','.join('?' * len(t.columns))})", t.rows)

val = [json.loads(x) for x in (run_dir / "data" / "val.jsonl").read_text(encoding="utf-8").splitlines() if x]
val = val[:n]


def norm_rows(rows, ordered):
    r = [tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows]
    return r if ordered else sorted(r, key=repr)


def clean(sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip(), flags=re.I)
    return sql.strip().rstrip(";").strip()


def score(preds):
    ok = err = exact = 0
    misses = []
    for ex, pred in zip(val, preds, strict=True):
        gold = ex["messages"][1]["content"]
        pred = clean(pred)
        exact += pred.lower().split() == gold.lower().split()
        try:
            got = norm_rows(db.execute(pred).fetchall(), "ORDER BY" in gold.upper())
        except sqlite3.Error:
            err += 1
            misses.append({"q": ex["messages"][0]["content"].split("\n")[0], "gold": gold, "pred": pred, "why": "error"})
            continue
        want = norm_rows(db.execute(gold).fetchall(), "ORDER BY" in gold.upper())
        if got == want:
            ok += 1
        elif len(misses) < 8:
            misses.append({"q": ex["messages"][0]["content"].split("\n")[0], "gold": gold, "pred": pred, "why": "wrong result"})
    k = len(val)
    return {"exec_accuracy": ok / k, "sql_error_rate": err / k, "exact_match": exact / k}, misses[:6]


b = HFBackend(run_dir)
model, tok, device = b._load()
res = {"n": len(val), "table": t.name, "table_rows": len(t.rows), "run_dir": str(run_dir)}
import os
INSTR = "Reply with exactly one SQLite query and nothing else (no explanation, no markdown).\n\n"
for name in os.environ.get("VARIANTS", "tuned,base").split(","):
    t0 = time.time()
    if name == "base_instructed":
        with model.disable_adapter():
            preds = [_gen(model, tok, INSTR + ex["messages"][0]["content"], device, 96) for ex in val]
    elif name == "base":
        with model.disable_adapter():
            preds = [_gen(model, tok, ex["messages"][0]["content"], device, 96) for ex in val]
    else:
        preds = [_gen(model, tok, ex["messages"][0]["content"], device, 96) for ex in val]
    m, misses = score(preds)
    res[name] = {**m, "seconds": round(time.time() - t0, 1), "sample_misses": misses}
    print(name, m, flush=True)
res["caveats"] = [
    "single table; val questions share templates and values with training questions (question-level split), so this "
    "measures learning the SQL task on this schema, not transfer to new tables",
    "execution accuracy compares result sets on the real table (unordered unless the gold SQL has ORDER BY)",
    "one run, greedy decoding, no seeds or confidence interval",
]
out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
