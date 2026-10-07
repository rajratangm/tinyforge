"""Task-level evaluation for fine-tunes: generation-based metrics, tuned vs base on held-out examples.

Loss improvements do not prove a model got better at a task; this scores actual outputs.
Currently ships a text-to-SQL task (normalised exact match + SQL validity checked with SQLite).
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path

import torch

from . import hardware
from .finetune import FTConfig, load_base, load_tokenizer
from .warnings import Level, Report

# ------------------------------------------------------------------ text-to-SQL task

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def sql_extract(raw: str) -> str:
    """Lenient extraction so a chatty base model is not penalised for formatting alone."""
    m = _FENCE.search(raw)
    text = m.group(1) if m else raw.split("\n\n")[0]
    return text.strip()


def sql_norm(s: str) -> str:
    s = s.strip().rstrip(";").strip().lower().replace('"', "'").replace("`", "")
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\s*([(),=<>])\s*", r"\1", s)


def sql_valid(query: str, schema: str) -> bool:
    """True if SQLite can plan the query against the given CREATE TABLE schema."""
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(schema)
        con.execute("EXPLAIN " + query.strip().rstrip(";"))
        return True
    except sqlite3.Error:
        return False
    finally:
        con.close()


_LITERAL = re.compile(r"'([^']*)'|\"([^\"]*)\"|\b(\d+(?:\.\d+)?)\b")
_MAX_VM_STEPS = 2_000_000  # abort runaway queries (cross joins on generated rows) instead of hanging the eval


def _literals(*queries: str) -> tuple[list[str], list[float]]:
    """String and numeric literals in the queries. Double-quoted strings count: SQLite treats "x" as a string
    when no column has that name, and most text-to-SQL gold queries use them."""
    texts, nums = [], []
    for q in queries:
        for single, double, n in _LITERAL.findall(q):
            if n:
                nums.append(float(n))
            elif single or double:
                texts.append(single or double)
    return texts, nums


def _populate(con: sqlite3.Connection, seed: int, texts: list[str], nums: list[float],
              rows: int = 40) -> None:
    """Fill every table with deterministic rows. Most values come from the literals the gold query mentions
    (so multi-condition WHERE clauses can select rows), the rest from a small filler pool so queries still
    discriminate. Execution accuracy is only as strong as this data: see the self-match check in the tests."""
    rng = random.Random(seed)
    lit_text = list(texts) + [f"{int(n) if n == int(n) else n}" for n in nums]
    lit_num = [int(n) if n == int(n) else n for n in nums]
    for t in texts:  # numeric-looking text literals (e.g. "40") can also populate numeric columns
        try:
            lit_num.append(float(t))
        except ValueError:
            pass
    filler_text = ["alpha", "beta", "gamma", "delta", "x", "y"]
    filler_num = [0, 1, 2, 5, 10, 25, 50, 100]
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tables:
        cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
        for _ in range(rows):
            vals = []
            for c in cols:
                ctype = (c[2] or "").upper()
                numeric = any(k in ctype for k in ("INT", "REAL", "NUM", "FLOA", "DOUB", "DEC"))
                lits, filler = (lit_num, filler_num) if numeric else (lit_text, filler_text)
                vals.append(rng.choice(lits) if lits and rng.random() < 0.7 else rng.choice(filler))
            con.execute(f'INSERT INTO "{t}" VALUES ({",".join("?" * len(cols))})', vals)


def _run(con: sqlite3.Connection, query: str) -> list[tuple] | None:
    steps = [0]

    def guard() -> int:
        steps[0] += 1
        return 1 if steps[0] > _MAX_VM_STEPS // 1000 else 0

    con.set_progress_handler(guard, 1000)
    try:
        return con.execute(query.strip().rstrip(";")).fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.set_progress_handler(None, 0)


def _canon(rows: list[tuple], ordered: bool) -> list:
    return [tuple(r) for r in rows] if ordered else sorted(tuple(map(repr, r)) for r in rows)


def sql_exec_match(pred: str, ref: str, schema: str, seeds: tuple[int, ...] = (0, 1, 2)) -> bool:
    """Execution accuracy: gold and predicted SQL return the same result on every generated database.
    Row order only matters when the gold query has ORDER BY; column order always matters. A gold query that
    errors or returns no rows on a database is skipped for that seed; if every seed is skipped the example
    counts as not matching (no evidence)."""
    texts, nums = _literals(ref)
    ordered = " order by " in " " + ref.lower() + " "
    compared = 0
    for seed in seeds:
        con = sqlite3.connect(":memory:")
        try:
            try:
                con.executescript(schema)
                _populate(con, seed, texts, nums)
            except sqlite3.Error:
                return False  # unusable schema (e.g. a table created twice): no evidence, not a crash
            want = _run(con, ref)
            if not want:
                continue
            got = _run(con, pred)
            if got is None:
                return False
            compared += 1
            if _canon(got, ordered) != _canon(want, ordered):
                return False
        finally:
            con.close()
    return compared > 0


def sql_score(user: str, pred_raw: str, ref: str) -> dict:
    schema = user.split("\n\n", 1)[1] if "\n\n" in user else ""
    pred = sql_extract(pred_raw)
    return {
        "strict_em": sql_norm(pred_raw) == sql_norm(ref),
        "lenient_em": sql_norm(pred) == sql_norm(ref),
        "valid": sql_valid(pred, schema) if schema else False,
        "exec_acc": sql_exec_match(pred, ref, schema) if schema else False,
    }


TASKS = {"sql": sql_score}


def sql_instruct(user: str) -> str:
    """The prompt a user would write for an untuned model: explicit instruction + output format."""
    question, _, schema = user.partition("\n\n")
    return ("Write a single SQL query that answers the question, given the schema. "
            f"Output only the SQL query, nothing else.\n\nSchema:\n{schema}\n\nQuestion: {question}")


# A fair baseline is the base model *told* the task, not one that was never asked to write SQL.
INSTRUCT = {"sql": sql_instruct}

# ------------------------------------------------------------------ generation + scoring


@torch.no_grad()
def _gen_batch(model, tok, prompts: list[str], device: str, max_new: int, bs: int = 8) -> list[str]:
    tok.padding_side = "left"
    outs: list[str] = []
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                         tokenize=False) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs += [tok.decode(o[enc.input_ids.size(1):], skip_special_tokens=True).strip() for o in out]
    return outs


def _rates(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    out = {k: sum(r[k] for r in rows) / n for k in ("strict_em", "lenient_em", "valid", "exec_acc")}
    out["ci95"] = {k: wilson_ci(sum(r[k] for r in rows), len(rows)) for k in ("lenient_em", "exec_acc")}
    return out


def wilson_ci(successes: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for a proportion (better than +/- sqrt(p(1-p)/n) at small n and extreme p)."""
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def evaluate_task(run_dir: Path, task: str = "sql", n: int = 100, min_gain_pts: float = 0.0,
                  max_new: int = 96, data_dir: Path | None = None) -> dict:
    from peft import PeftModel

    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {list(TASKS)}")
    score = TASKS[task]
    cfg = FTConfig.model_validate_json((run_dir / "ft_config.json").read_text())
    data_dir = data_dir or cfg.data_dir
    hw = hardware.probe()
    tok = load_tokenizer(cfg.base_model)
    recs = [json.loads(line)["messages"] for line in
            (data_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()][:n]
    users = [next(m["content"] for m in r if m["role"] == "user") for r in recs]
    refs = [r[-1]["content"] for r in recs]

    model = load_base(cfg.base_model, cfg.quant, hw.bf16_supported, hw.device)
    model = PeftModel.from_pretrained(model, run_dir / "best")
    model.eval()
    tuned_out = _gen_batch(model, tok, users, hw.device, max_new)
    with model.disable_adapter():
        raw_out = _gen_batch(model, tok, users, hw.device, max_new)
        instr_out = _gen_batch(model, tok, [INSTRUCT[task](u) for u in users], hw.device, max_new)

    def rows(outs):
        return [score(u, p, r) for u, p, r in zip(users, outs, refs, strict=True)]

    tr, br_raw, br_instr = _rates(rows(tuned_out)), _rates(rows(raw_out)), _rates(rows(instr_out))
    best_base = max((br_raw, br_instr), key=lambda x: x["lenient_em"])
    gain_pts = 100 * (tr["lenient_em"] - best_base["lenient_em"])
    res = {"task": task, "n": len(recs), "tuned": tr, "base_raw_prompt": br_raw,
           "base_instructed": br_instr, "gain_pts_lenient_em": gain_pts,
           "relative_gain": (tr["lenient_em"] / best_base["lenient_em"] - 1)
           if best_base["lenient_em"] else None,
           "samples": [{"question": u.split("\n\n")[0], "reference": r, "base_raw": b,
                        "base_instructed": i, "tuned": t}
                       for u, r, b, i, t in list(zip(users, refs, raw_out, instr_out, tuned_out,
                                                     strict=True))[:6]]}

    rep = Report()
    if len(recs) < 50:
        rep.add("TK005", Level.WARN, f"Only {len(recs)} eval examples: +/-{100 / len(recs) ** 0.5:.0f} "
                "points of noise.", "Use --n 100 or more.")
    if tr["lenient_em"] <= best_base["lenient_em"]:
        rep.add("TK001", Level.ERROR, f"Tuned exact-match {tr['lenient_em']:.1%} is not better than the "
                f"best base prompt ({best_base['lenient_em']:.1%}).")
    elif gain_pts < min_gain_pts:
        rep.add("TK002", Level.ERROR, f"Gain {gain_pts:.1f} points over the best base prompt is below "
                f"the required {min_gain_pts:.1f}.")
    if tr["valid"] < 0.9:
        rep.add("TK003", Level.WARN, f"Only {tr['valid']:.0%} of tuned outputs are valid SQL.")
    rep.add("TK004", Level.INFO, "Gain is measured against the stronger of two base prompts "
            "(raw vs. explicit instruction), so it is not inflated by an untold base model.")
    res["diagnostics"] = rep.to_list()
    res["passed"] = not rep.has_errors
    (run_dir / "task_eval.json").write_text(json.dumps(res, indent=2))
    return res
