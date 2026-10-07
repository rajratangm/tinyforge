"""Benchmark catalog, recommender, harness command builder, built-in SQL check. No GPU, no network."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import pytest
from typer.testing import CliRunner

from tinyforge import bench, sqleval
from tinyforge.cli import app
from tinyforge.memtiers import Hierarchy, Tier


def hier(vram_gb=4.0, vram_bw=150.0, ram_bw=16.0) -> Hierarchy:
    vram = Tier("vram", "gpu", vram_gb, vram_gb * 0.8, vram_bw, True) if vram_gb else None
    return Hierarchy(
        vram,
        Tier("ram", "ddr4", 16, 6, ram_bw, True),
        Tier("disk", "nvme", 900, 200, 1.5, True),
        {},
        0.0,
        10.0,
        "test",
    )


def by_name(recs):
    return {r.name: r for r in recs}


def test_catalog_is_consistent():
    for name, b in bench.CATALOG.items():
        assert b.name == name and b.kind in ("loglik", "generate") and b.items > 0 and b.goals
        assert (b.kind == "loglik") == (b.decode_tokens == 0)


def test_decode_speed_model_matches_the_measured_case_in_order_of_magnitude():
    tps, why = bench.estimate_decode_tps(hier(), 8.0, "4bit")  # 8B Q4 on a 4 GB card: measured 10-18 tok/s
    assert 6 <= tps <= 25 and "GPU" in why
    small, _ = bench.estimate_decode_tps(hier(), 0.36, "4bit")
    assert small > tps * 5  # a small model that fits entirely on the GPU is much faster
    cpu, why = bench.estimate_decode_tps(hier(vram_gb=0), 8.0, "4bit")
    assert cpu < tps and "CPU only" in why


def test_8b_on_4gb_can_only_run_generative_benchmarks_and_says_why():
    recs, ctx = bench.recommend(hier(), 8.0, "4bit", minutes=30, goal="general")
    r = by_name(recs)
    assert not ctx["model_fits_gpu"]
    assert not r["mmlu"].runnable and "resident" in r["mmlu"].why[0]
    assert r["gsm8k"].runnable and r["ifeval"].runnable and r["gsm8k"].limit > 0


def test_small_model_that_fits_unlocks_loglikelihood_benchmarks():
    recs, ctx = bench.recommend(hier(), 1.5, "4bit", minutes=30, goal="forgetting")
    r = by_name(recs)
    assert ctx["model_fits_gpu"] and r["mmlu"].runnable and r["arc_easy"].runnable
    server, _ = bench.recommend(hier(), 8.0, "4bit", minutes=30, goal="forgetting", server_logprobs=True)
    assert by_name(server)["mmlu"].runnable  # a server that returns logprobs lifts the restriction


def test_budget_scales_samples_and_never_exceeds_the_benchmark_size():
    short, _ = bench.recommend(hier(), 8.0, "4bit", minutes=10, goal="instruction")
    long_, _ = bench.recommend(hier(), 8.0, "4bit", minutes=120, goal="instruction")
    assert by_name(long_)["ifeval"].limit > by_name(short)["ifeval"].limit
    assert by_name(long_)["ifeval"].limit <= bench.CATALOG["ifeval"].items
    tiny, _ = bench.recommend(hier(), 8.0, "4bit", minutes=0.1, goal="instruction")
    r = by_name(tiny)["ifeval"]
    assert r.limit >= 10 and any("very small sample" in w for w in r.why)  # warns instead of hiding the noise
    assert sum(x.est_minutes for x in short if x.name in ("ifeval",)) <= 10 * 1.5


def test_goal_filters_and_measured_speed_overrides_the_estimate():
    recs, ctx = bench.recommend(hier(), 8.0, "4bit", minutes=30, goal="sql", measured_tps=40.0)
    assert ctx["decode_tps"] == 40.0 and "measured" in ctx["decode_tps_basis"]
    r = by_name(recs)
    assert r["sql-exec"].runnable and "not a 'sql' benchmark" in " ".join(r["gsm8k"].why)


def test_lm_eval_commands_for_server_and_hf_targets():
    chat = bench.lm_eval_cmd(bench.CATALOG["gsm8k"], 50, server_url="http://x:1/", model_name="m")
    assert "local-chat-completions" in chat and "--apply_chat_template" in chat and "50" in chat
    assert any("base_url=http://x:1/v1/chat/completions" in a for a in chat)
    ll = bench.lm_eval_cmd(bench.CATALOG["arc_easy"], 20, server_url="http://x:1")
    assert "local-completions" in ll and any("/v1/completions" in a for a in ll)
    hf = bench.lm_eval_cmd(bench.CATALOG["mmlu"], 30, hf_model="models/m", peft="runs/a/best")
    assert "hf" in hf and any("peft=runs/a/best" in a and "load_in_4bit=True" in a for a in hf)
    with pytest.raises(ValueError, match="built-in"):
        bench.lm_eval_cmd(bench.CATALOG["sql-exec"], 10, server_url="http://x")
    with pytest.raises(ValueError, match="either"):
        bench.lm_eval_cmd(bench.CATALOG["gsm8k"], 10)


def test_run_lm_eval_parses_results_and_reports_missing_dependency(tmp_path):
    out = tmp_path / "o" / "m"
    out.mkdir(parents=True)
    (out / "results_2026.json").write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict": 0.5}}}))
    ok = bench.run_lm_eval(
        ["x"], str(tmp_path / "o"), runner=lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")
    )
    assert ok["exit"] == 0 and ok["results"]["gsm8k"]["exact_match,strict"] == 0.5
    miss = bench.run_lm_eval(
        ["x"],
        str(tmp_path / "none"),
        runner=lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "No module named lm_eval"),
    )
    assert miss["exit"] == 1 and "lm-eval" in miss["missing_dependency"] and miss["results"] == {}


def test_cli_list_suggest_and_dry_run(tmp_path):
    r = CliRunner()
    names = [b["name"] for b in json.loads(r.invoke(app, ["bench", "list", "--json"]).stdout)]
    assert "mmlu" in names and "sql-exec" in names
    s = r.invoke(
        app,
        [
            "bench",
            "suggest",
            "--params-b",
            "1.5",
            "--no-measure",
            "--goal",
            "forgetting",
            "--measured-tps",
            "40",
            "--json",
        ],
    )
    out = json.loads(s.stdout)
    assert out["context"]["decode_tps"] == 40.0 and any(x["name"] == "mmlu" for x in out["recommendations"])
    d = r.invoke(
        app,
        [
            "bench",
            "run",
            "gsm8k",
            "arc_easy",
            "--server-url",
            "http://127.0.0.1:1",
            "--limit",
            "5",
            "--out",
            str(tmp_path / "b"),
            "--dry-run",
            "--json",
        ],
    )
    runs = json.loads(d.stdout)["runs"]
    assert d.exit_code == 0 and "lm_eval" in runs["gsm8k/model"]["dry_run"] and not (tmp_path / "b").exists()
    cmp_ = r.invoke(
        app, ["bench", "run", "gsm8k", "--server-url", "http://x", "--compare", "--dry-run", "--json"]
    )
    assert set(json.loads(cmp_.stdout)["runs"]) == {"gsm8k/tuned", "gsm8k/base"}


def test_cli_rejects_unknown_benchmarks_and_incomplete_sql_exec():
    r = CliRunner()
    assert r.invoke(app, ["bench", "run", "nope"]).exit_code == 2
    assert r.invoke(app, ["bench", "run", "sql-exec", "--server-url", "http://x"]).exit_code == 2


def test_sql_exec_scores_result_sets_not_strings(tmp_path):
    (tmp_path / "t.csv").write_text("city,wins\nParis,3\nRome,5\nParis,4\n")
    test = [
        {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "SELECT SUM(wins) FROM t"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "SELECT city FROM t"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "SELECT COUNT(*) FROM t"},
            ]
        },
    ]
    (tmp_path / "test.jsonl").write_text("\n".join(json.dumps(x) for x in test), encoding="utf-8")
    answers = {
        "q1": "SELECT  sum(wins)  FROM t;",
        "q2": "```sql\nSELECT city FROM t\n```",
        "q3": "I think it is 3",
    }
    scales: list[float] = []
    res = sqleval.run(
        "http://x",
        tmp_path / "t.csv",
        tmp_path / "test.jsonl",
        10,
        ("tuned", "base_instructed"),
        ask=lambda url, p: answers[p.split("\n\n")[-1]],
        set_scale=lambda u, s: scales.append(s),
    )
    assert scales == [1.0, 0.0] and res["n"] == 3
    # q1 differs only in case/spacing and q2 only in fences: both equal by result; q3 is prose -> SQL error
    assert res["tuned"]["exec_accuracy"] == pytest.approx(2 / 3) and res["tuned"][
        "sql_error_rate"
    ] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="unknown variant"):
        sqleval.run(
            "http://x",
            tmp_path / "t.csv",
            tmp_path / "test.jsonl",
            1,
            ("nope",),
            ask=lambda *a: "",
            set_scale=lambda *a: None,
        )
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE t (a INTEGER)")
    db.executemany("INSERT INTO t VALUES (?)", [(2,), (1,)])
    ordered = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "SELECT a FROM t ORDER BY a"},
            ]
        }
    ]
    assert (
        sqleval.score(db, ordered, ["SELECT a FROM t"])[0]["exec_accuracy"] == 0.0
    )  # ORDER BY: order matters
    assert sqleval.score(db, ordered[:1], ["SELECT a FROM t ORDER BY a"])[0]["exec_accuracy"] == 1.0
