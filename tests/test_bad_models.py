"""The tool must tell the truth about bad models: untrained bases, looping text, no chat template."""

from __future__ import annotations

import json

import pytest

from tinyforge import cli, ft_data, ft_eval, worker
from tinyforge.errors import UserError
from tinyforge.finetune import ChatDataset, require_chat_template


def test_compression_ratio_separates_looping_text_from_prose():
    looping = "clandestine " * 30
    # a real output of the random model (see benchmarks/bad-model-random-smollm2.json)
    symbols = "SELECT FROM table_ FROM table_name_" + "1_" * 25 + "_" * 36
    prose = "A hash table maps keys to values using a hash function, which gives fast average lookup time."
    assert ft_eval.compress_ratio(looping) < ft_eval.DEGENERATE_RATIO
    assert ft_eval.compress_ratio(symbols) < ft_eval.DEGENERATE_RATIO
    assert ft_eval.compress_ratio(prose) > ft_eval.DEGENERATE_RATIO


class NoTemplate:
    chat_template = None


class WithTemplate:
    chat_template = "{{ messages }}"


def test_missing_chat_template_is_a_user_error_with_a_fix(tmp_path):
    with pytest.raises(UserError, match="no chat template.*Instruct"):
        require_chat_template(NoTemplate(), "tiny-gpt2")
    require_chat_template(WithTemplate())  # fine
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"messages": []}))
    with pytest.raises(UserError):
        ChatDataset(p, NoTemplate(), 64)


def test_ft_data_reports_fd007_instead_of_crashing(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: NoTemplate())
    src = tmp_path / "s.jsonl"
    src.write_text(json.dumps({"instruction": "q", "output": "a"}))
    meta, rep = ft_data.prepare(str(src), tmp_path / "o", "gpt2-like")
    assert rep.has_errors and rep.items[0].code == "FD007" and "Instruct" in rep.items[0].fix
    assert meta["train"] == 0


def test_main_turns_user_errors_into_a_plain_message_and_exit_2(monkeypatch, capsys):
    def boom():
        raise UserError("use an instruct model")

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "use an instruct model" in err and "Traceback" not in err


def test_relative_gain_gates_fail_closed_when_the_base_model_is_untrained(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ft_eval,
        "evaluate",
        lambda *a, **k: {"base_untrained": True, "improvement_pct": 40.0, "forgetting_pct": 0.0},
    )
    events = []
    ok = worker.evaluate_gates(
        [{"metric": "gain_pct", "op": ">=", "value": 5}],
        {"best_val_loss": 1.0},
        tmp_path,
        "text",
        lambda kind, **kw: events.append((kind, kw)),
    )
    assert ok is False
    kind, kw = events[0]
    assert kind == "gate" and kw["passed"] is False and "untrained" in kw["reason"]
