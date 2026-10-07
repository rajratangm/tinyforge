from __future__ import annotations

import json
import sqlite3

from typer.testing import CliRunner

from tinyforge.cli import app
from tinyforge.tabular import generate, load_csv

CSV = """Team Name,City,Wins,Score avg
Ants,Paris,10,3.5
Bees,Paris,7,2.5
Cats,Rome,12,4.0
Dogs,Rome,3,1.0
Emus,Oslo,9,3.0
"""


def _table(tmp_path):
    p = tmp_path / "teams.csv"
    p.write_text(CSV)
    return load_csv(p)


def test_load_csv_sanitizes_and_infers_types(tmp_path):
    t = _table(tmp_path)
    assert t.columns == ["team_name", "city", "wins", "score_avg"]
    assert t.kinds == {"team_name": "text", "city": "text", "wins": "int", "score_avg": "float"}
    assert t.name == "teams" and len(t.rows) == 5


def test_every_generated_sql_executes_and_matches_schema(tmp_path):
    t = _table(tmp_path)
    ex, stats = generate(t, count=200, seed=1)
    assert ex and stats["failed_to_execute"] == 0
    db = sqlite3.connect(":memory:")
    db.execute(t.create_sql())
    db.executemany("INSERT INTO teams VALUES (?,?,?,?)", t.rows)
    for e in ex:
        sql = e["messages"][1]["content"]
        assert db.execute(sql).fetchall() is not None
        assert t.create_sql() in e["messages"][0]["content"]


def test_ambiguous_top_questions_are_dropped(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("grp,val\na,5\nb,5\na,1\nb,2\n")  # tie at the top value
    ex, _ = generate(load_csv(p), count=500)
    assert not any("ORDER BY" in e["messages"][1]["content"] for e in ex)


def test_quotes_in_values_are_escaped(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("name,kind\nO'Brien,x\nSmith,y\nO'Brien,x\n")
    ex, stats = generate(load_csv(p), count=100)
    assert stats["failed_to_execute"] == 0
    assert any("O''Brien" in e["messages"][1]["content"] for e in ex)


def test_cli_writes_jsonl(tmp_path):
    p = tmp_path / "teams.csv"
    p.write_text(CSV)
    out = tmp_path / "o.jsonl"
    res = CliRunner().invoke(app, ["data", "tabular", str(p), "--out", str(out), "--count", "20", "--json"])
    assert json.loads(res.stdout)["meta"]["verified_by"] == "sqlite execution"
    assert 0 < len(out.read_text().splitlines()) <= 20
