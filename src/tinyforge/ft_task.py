"""Task-level evaluation for fine-tunes: generation-based metrics, tuned vs base on held-out examples.

Loss improvements do not prove a model got better at a task; this scores actual outputs.
Currently ships a text-to-SQL task (normalised exact match + SQL validity checked with SQLite).
"""

from __future__ import annotations

import json
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


def sql_score(user: str, pred_raw: str, ref: str) -> dict:
    schema = user.split("\n\n", 1)[1] if "\n\n" in user else ""
    pred = sql_extract(pred_raw)
    return {
        "strict_em": sql_norm(pred_raw) == sql_norm(ref),
        "lenient_em": sql_norm(pred) == sql_norm(ref),
        "valid": sql_valid(pred, schema) if schema else False,
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
    return {k: sum(r[k] for r in rows) / n for k in ("strict_em", "lenient_em", "valid")}


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
