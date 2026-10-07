from __future__ import annotations

import json

from tinyforge.pairs import generate, grounded, run, split_sources

CHUNK = ("The Apollo 11 mission landed the first humans on the Moon in 1969. Neil Armstrong and Buzz Aldrin "
         "spent about two and a half hours outside the lunar module while Michael Collins orbited above.")


def _teacher(reply: str):
    return lambda prompt: reply


def test_grounded_accepts_supported_and_rejects_invented():
    assert grounded("the first humans on the Moon in 1969", CHUNK)
    assert not grounded("Yuri Gagarin walked on Mars in 1985", CHUNK)
    assert not grounded("", CHUNK)


def test_split_is_by_source_and_deterministic():
    rows = [{"source": f"f{i % 20}.txt", "text": str(i)} for i in range(200)]
    tr, va = split_sources(rows, 30)
    assert {r["source"] for r in tr}.isdisjoint({r["source"] for r in va})
    assert (tr, va) == split_sources(rows, 30) and va


def test_generate_keeps_grounded_drops_rest_and_counts():
    reply = json.dumps([
        {"question": "Who orbited above the lunar module?", "answer": "Michael Collins orbited above"},
        {"question": "Who first walked on Mars?", "answer": "Yuri Gagarin walked on Mars"},
        {"question": "Who orbited above the lunar module?", "answer": "Michael Collins"},
        {"question": "Why?", "answer": "x"},
    ])
    ex, st = generate([{"source": "a", "text": CHUNK}], _teacher("Sure!\n" + reply), per_chunk=4)
    assert (st["kept"], st["ungrounded"], st["duplicate_question"], st["too_short"]) == (1, 1, 1, 1)
    assert CHUNK in ex[0]["messages"][0]["content"]


def test_bad_teacher_output_and_errors_do_not_crash():
    def boom(_):
        raise RuntimeError("down")

    _, st = generate([{"source": "a", "text": CHUNK}], boom)
    assert st["teacher_errors"] == 1
    _, st = generate([{"source": "a", "text": CHUNK}], _teacher("not json at all"))
    assert st["unparseable"] == 1


def test_run_writes_split_and_meta(tmp_path):
    rows = [{"text": CHUNK, "source": f"doc{i}.txt", "index": 0} for i in range(30)]
    cp = tmp_path / "c.jsonl"
    cp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    reply = json.dumps([{"question": "Who orbited above the lunar module?", "answer": "Michael Collins"}])
    meta = run(cp, tmp_path / "o", _teacher(reply), "fake", per_chunk=1, val_pct=20)
    assert meta["teacher"] == "fake" and (tmp_path / "o" / "meta.json").exists()
    assert meta["train"]["kept"] >= 1


def test_parser_recovers_pairs_from_fenced_truncated_and_chatty_replies():
    from tinyforge.pairs import _parse

    full = '[{"question": "Who orbited?", "answer": "Collins"}, {"question": "When?", "answer": "1969"}]'
    assert len(_parse(full)) == 2
    assert len(_parse("Sure! Here you go:\n```json\n" + full + "\n```\nHope that helps.")) == 2
    cut = '[{"question": "Who orbited?", "answer": "Collins"}, {"question": "When did it hap'
    assert [p["answer"] for p in _parse(cut)] == ["Collins"]  # complete objects survive a cut-off array
    lines = '{"question": "A one?", "answer": "x"}\n{"question": "B two?", "answer": "y"}'
    assert len(_parse(lines)) == 2
    assert _parse("I cannot do that.") == [] and _parse('[{"question": 5, "answer": "x"}]') == []


def test_unparseable_replies_are_sampled_in_stats():
    _, st = generate([{"source": "a", "text": CHUNK}], _teacher("no json here at all"))
    assert st["unparseable"] == 1 and st["unparseable_samples"] == ["no json here at all"]
