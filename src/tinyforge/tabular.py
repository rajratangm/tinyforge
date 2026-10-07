"""Tabular data -> verified text-to-SQL training examples.

A CSV is loaded into an in-memory SQLite table; questions come from templates over real values, and every
SQL answer is EXECUTED before it is kept (errors, wrong-shaped and ambiguous results are dropped). Output uses
the chat format of `ft_data` (user = question + CREATE TABLE, assistant = SQL), so the SQL exec-accuracy
harness consumes it unchanged.

What this teaches: querying a table through SQL. It does not teach a model facts from the rows. Questions over
one table share values, so a question-level train/val split measures templates-on-seen-data, not
generalisation to new schemas; for that, hold out whole tables.
"""
from __future__ import annotations

import csv
import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MAX_CATEGORIES = 30      # text columns with more distinct values are free text and are not filtered on


@dataclass
class Table:
    name: str
    columns: list[str]
    kinds: dict[str, str]          # column -> "int" | "float" | "text"
    rows: list[tuple]

    def create_sql(self) -> str:
        typ = {"int": "INTEGER", "float": "REAL", "text": "TEXT"}
        cols = ", ".join(f"{c} {typ[self.kinds[c]]}" for c in self.columns)
        return f"CREATE TABLE {self.name} ({cols})"


def _ident(s: str, used: set[str]) -> str:
    out = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip().lower()).strip("_") or "col"
    if out[0].isdigit():
        out = "c_" + out
    base, i = out, 2
    while out in used:
        out, i = f"{base}_{i}", i + 1
    used.add(out)
    return out


def _kind(values: list[str]) -> str:
    vals = [v for v in values if v != ""]
    if not vals:
        return "text"
    for cast, kind in ((int, "int"), (float, "float")):
        try:
            [cast(v) for v in vals]
            return kind
        except ValueError:
            pass
    return "text"


def load_csv(path: Path, name: str | None = None) -> Table:
    with path.open(newline="", encoding="utf-8-sig") as f:
        raw = list(csv.reader(f))
    if len(raw) < 2:
        raise ValueError(f"{path.name}: need a header and at least one row")
    used: set[str] = set()
    cols = [_ident(h, used) for h in raw[0]]
    body = [r for r in raw[1:] if len(r) == len(cols)]
    kinds = {c: _kind([r[i] for r in body]) for i, c in enumerate(cols)}
    conv = {"int": int, "float": float, "text": str}
    rows = [tuple(None if r[i] == "" else conv[kinds[c]](r[i]) for i, c in enumerate(cols)) for r in body]
    return Table(_ident(name or path.stem, set()), cols, kinds, rows)


def _lit(v) -> str:
    return "'" + v.replace("'", "''") + "'" if isinstance(v, str) else repr(v)


def _q(c: str) -> str:
    return c.replace("_", " ")


def _candidates(t: Table, rng: random.Random):
    """Yield (question, sql, min_rows, max_rows) over real values; execution decides what survives."""
    cats = [c for c in t.columns if t.kinds[c] == "text"
            and 1 < len({r[t.columns.index(c)] for r in t.rows}) <= MAX_CATEGORIES]
    nums = [c for c in t.columns if t.kinds[c] in ("int", "float")]
    idx = {c: i for i, c in enumerate(t.columns)}
    n = t.name
    yield f"How many rows are in {n}?", f"SELECT COUNT(*) FROM {n}", 1, 1
    for r in rng.sample(t.rows, len(t.rows)):  # every row once, shuffled: covers values, no repeats
        for c in cats:
            v = r[idx[c]]
            if v is None:
                continue
            phrasing = [f"How many rows have {_q(c)} equal to {v}?", f"Count the rows where {_q(c)} is {v}."]
            yield (rng.choice(phrasing),
                   f"SELECT COUNT(*) FROM {n} WHERE {c} = {_lit(v)}", 1, 1)
            for o in t.columns:
                if o != c and r[idx[o]] is not None:
                    yield (f"List the {_q(o)} where {_q(c)} is {v}.",
                           f"SELECT {o} FROM {n} WHERE {c} = {_lit(v)}", 1, 15)
            for m in nums:
                for fn, word in (("MAX", "highest"), ("MIN", "lowest"), ("AVG", "average"), ("SUM", "total")):
                    yield (f"What is the {word} {_q(m)} where {_q(c)} is {v}?",
                           f"SELECT {fn}({m}) FROM {n} WHERE {c} = {_lit(v)}", 1, 1)
        for m in nums:
            v = r[idx[m]]
            if v is not None:
                yield (f"How many rows have {_q(m)} greater than {v}?",
                       f"SELECT COUNT(*) FROM {n} WHERE {m} > {_lit(v)}", 1, 1)
    for c in cats:
        yield (f"How many rows are there for each {_q(c)}?",
               f"SELECT {c}, COUNT(*) FROM {n} GROUP BY {c}", 2, 60)
        for m in nums:
            yield (f"Which {_q(c)} has the highest {_q(m)}?",
                   f"SELECT {c} FROM {n} ORDER BY {m} DESC LIMIT 1", 1, 1)


def generate(t: Table, count: int = 500, seed: int = 0) -> tuple[list[dict], dict]:
    """Return (examples, stats). Every kept example's SQL ran without error and returned a sensible result."""
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.execute(t.create_sql())
    db.executemany(f"INSERT INTO {t.name} VALUES ({','.join('?' * len(t.columns))})", t.rows)
    schema = t.create_sql()
    seen: set[str] = set()
    out: list[dict] = []
    tried = failed = bad_shape = 0
    for q, sql, lo, hi in _candidates(t, rng):
        if q in seen:
            continue
        seen.add(q)
        tried += 1
        try:
            res = db.execute(sql).fetchall()
        except sqlite3.Error:
            failed += 1
            continue
        if not (lo <= len(res) <= hi) or (len(res) == 1 and res[0] == (None,)):
            bad_shape += 1
            continue
        if "ORDER BY" in sql and not _unique_extreme(db, t, sql):
            bad_shape += 1
            continue
        if len(out) >= count * 2:  # enough to shuffle and cut; avoids enumerating huge tables
            break
        out.append({"messages": [{"role": "user", "content": f"{q}\n\n{schema}"},
                                 {"role": "assistant", "content": sql}]})
    rng.shuffle(out)
    stats = {"candidates": tried, "failed_to_execute": failed, "dropped_bad_shape": bad_shape,
             "kept": min(count, len(out)), "available": len(out), "table": t.name, "rows": len(t.rows),
             "verified_by": "sqlite execution",
             "split_note": "question-level only; hold out tables to test transfer"}
    return out[:count], stats


def _unique_extreme(db: sqlite3.Connection, t: Table, sql: str) -> bool:
    """A 'highest X' question is only well-posed if the top value is not tied."""
    m = re.search(r"ORDER BY (\w+) DESC LIMIT 1", sql)
    if not m:
        return True
    top = db.execute(f"SELECT {m.group(1)} FROM {t.name} ORDER BY {m.group(1)} DESC LIMIT 2").fetchall()
    return len(top) == 1 or top[0] != top[1]


def _hard_candidates(t: Table, rng: random.Random):
    """Question shapes that `_candidates` never produces: use them for held-out evaluation, not training."""
    idx = {c: i for i, c in enumerate(t.columns)}
    cats = [c for c in t.columns if t.kinds[c] == "text"
            and 1 < len({r[idx[c]] for r in t.rows}) <= MAX_CATEGORIES]
    nums = [c for c in t.columns if t.kinds[c] in ("int", "float")]
    n = t.name
    for c in cats:
        yield f"List the distinct values of {_q(c)}.", f"SELECT DISTINCT {c} FROM {n}", 2, MAX_CATEGORIES
        yield f"How many different {_q(c)} values are there?", f"SELECT COUNT(DISTINCT {c}) FROM {n}", 1, 1
        counts = sorted(Counter(r[idx[c]] for r in t.rows if r[idx[c]] is not None).values())
        if len(counts) > 2:
            mid = counts[len(counts) // 2]
            yield (f"Which {_q(c)} values appear in more than {mid} rows?",
                   f"SELECT {c} FROM {n} GROUP BY {c} HAVING COUNT(*) > {mid}", 1, MAX_CATEGORIES)
        for m in nums:
            yield (f"What is the average {_q(m)} for each {_q(c)}?",
                   f"SELECT {c}, AVG({m}) FROM {n} GROUP BY {c}", 2, MAX_CATEGORIES)
    for m in nums:
        vals = sorted(r[idx[m]] for r in t.rows if r[idx[m]] is not None)
        if len(vals) < 10:
            continue
        lo, hi = vals[len(vals) // 4], vals[3 * len(vals) // 4]
        yield (f"How many rows have {_q(m)} between {lo} and {hi}?",
               f"SELECT COUNT(*) FROM {n} WHERE {m} BETWEEN {_lit(lo)} AND {_lit(hi)}", 1, 1)
        yield (f"What are the 3 highest {_q(m)} values?",
               f"SELECT {m} FROM {n} ORDER BY {m} DESC LIMIT 3", 3, 3)
    for _ in range(len(t.rows)):
        r = rng.choice(t.rows)
        if len(cats) >= 2:
            c1, c2 = rng.sample(cats, 2)
            v1, v2 = r[idx[c1]], r[idx[c2]]
            if v1 is not None and v2 is not None:
                yield (f"How many rows have {_q(c1)} equal to {v1} and {_q(c2)} equal to {v2}?",
                       f"SELECT COUNT(*) FROM {n} WHERE {c1} = {_lit(v1)} AND {c2} = {_lit(v2)}", 1, 1)
        if cats and nums:
            c, m = rng.choice(cats), rng.choice(nums)
            vc, vm = r[idx[c]], r[idx[m]]
            if vc is not None and vm is not None:
                yield (f"How many rows have {_q(c)} equal to {vc} and {_q(m)} greater than {vm}?",
                       f"SELECT COUNT(*) FROM {n} WHERE {c} = {_lit(vc)} AND {m} > {_lit(vm)}", 1, 1)


def generate_hard(t: Table, count: int = 100, seed: int = 0) -> tuple[list[dict], dict]:
    """Verified examples of harder shapes (AND, BETWEEN, DISTINCT, HAVING, GROUP BY AVG, top-N). Eval only."""
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.execute(t.create_sql())
    db.executemany(f"INSERT INTO {t.name} VALUES ({','.join('?' * len(t.columns))})", t.rows)
    schema, seen, out = t.create_sql(), set(), []
    failed = bad = 0
    for q, sql, lo, hi in _hard_candidates(t, rng):
        if q in seen:
            continue
        seen.add(q)
        try:
            res = db.execute(sql).fetchall()
        except sqlite3.Error:
            failed += 1
            continue
        trivial = len(res) == 1 and len(res[0]) == 1 and res[0][0] in (None, 0)
        if not (lo <= len(res) <= hi) or trivial:
            bad += 1
            continue
        out.append({"messages": [{"role": "user", "content": f"{q}\n\n{schema}"},
                                 {"role": "assistant", "content": sql}]})
    rng.shuffle(out)
    return out[:count], {"available": len(out), "failed_to_execute": failed, "dropped_bad_shape": bad,
                         "table": t.name, "verified_by": "sqlite execution"}
