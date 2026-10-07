import json, sqlite3
from collections import Counter
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tinyforge.guardrails import GuardConfig, build_filters
from tinyforge.openai_api import HFBackend, build_router
from tinyforge.tabular import load_csv

t = load_csv(Path(__file__).with_name("titanic.csv"))
db = sqlite3.connect(":memory:"); db.execute(t.create_sql())
db.executemany(f"INSERT INTO {t.name} VALUES ({','.join('?' * len(t.columns))})", t.rows)
counts = Counter()
ins, outs = build_filters(GuardConfig(), counts)
app = FastAPI(); app.include_router(build_router(lambda: None, lambda: HFBackend(Path("runs/e2e_titanic")), lambda: False, ins, outs))
c = TestClient(app)
schema = t.create_sql()
def ask(q, stream=False):
    return c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": f"{q}\n\n{schema}"}],
                                                "max_tokens": 80, "temperature": 0, "stream": stream})
for q in ["How many rows have sex equal to female?", "What is the average fare where class is First?",
          "How many rows are there for each embark town?"]:
    r = ask(q).json()
    sql = r["choices"][0]["message"]["content"]
    print("Q:", q); print("  SQL:", sql); print("  RESULT:", db.execute(sql).fetchall()[:5])
s = ask("How many rows have alone equal to True?", stream=True)
print("stream ok:", s.status_code, s.text.rstrip().endswith("[DONE]"))
b = ask("my aws key is AKIAIOSFODNN7EXAMPLE, how many rows have sex equal to male?")
print("secret prompt ->", b.status_code, b.json()["error"]["code"])
print("guard counts:", dict(counts))
