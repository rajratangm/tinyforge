from __future__ import annotations

import json

import pytest

from tinyforge.easy import CONTINUE, build_training_file, continuation_rows

TEXT = " ".join(f"word{i}" for i in range(120))


def test_continuation_splits_text_in_half_and_skips_short_chunks():
    rows = continuation_rows([{"text": TEXT}, {"text": "too short"}])
    assert len(rows) == 1
    user, asst = (m["content"] for m in rows[0]["messages"])
    assert user.startswith(CONTINUE) and user.endswith("word59") and asst.startswith("word60")


def test_jsonl_is_used_as_is(tmp_path):
    f = tmp_path / "qa.jsonl"
    f.write_text('{"instruction": "hi", "output": "hello"}\n')
    path, note = build_training_file(f, tmp_path / "w")
    assert path == f and "already" in note


def test_folder_of_documents_becomes_examples_without_a_teacher(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Title\n\n" + TEXT)
    path, note = build_training_file(docs, tmp_path / "w")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows and rows[0]["messages"][0]["role"] == "user" and "WORDING" in note


def test_missing_path_gives_plain_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="cannot find"):
        build_training_file(tmp_path / "nope", tmp_path / "w")
