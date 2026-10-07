from __future__ import annotations

import json

from typer.testing import CliRunner

from tinyforge.cli import app
from tinyforge.etl import Chunk, chunk_text, clean, dedupe, extract, ingest

A = ("The water cycle describes how water evaporates from the surface of the earth rises into the atmosphere "
     "cools and condenses into clouds and falls again as precipitation some flows over land as runoff")
B = ("Binary search finds a target value in a sorted array by repeatedly dividing the search interval "
     "in half and comparing the middle element with the target until the value is found or it is empty")


def test_clean_rejoins_hyphenation_and_drops_repeated_headers():
    hdr = "ACME Corp Confidential\n"
    text = hdr + "first para-\ngraph here\n" + hdr + "body\n" + hdr
    out = clean(text)
    assert "paragraph" in out and "ACME" not in out


def test_chunking_respects_limit_and_paragraphs():
    text = "\n\n".join(" ".join(["w"] * 40) for _ in range(5))
    chunks = chunk_text(text, max_words=100)
    assert all(len(c.split()) <= 100 for c in chunks) and len(chunks) == 3
    assert len(chunk_text(" ".join(["x"] * 250), max_words=100)) == 3


def test_dedupe_exact_near_and_distinct():
    near = A.replace("runoff", "run off water")  # small edit
    kept, exact, nr = dedupe([Chunk(A, "a", 0), Chunk(A, "b", 0), Chunk(near, "c", 0), Chunk(B, "d", 0)])
    assert exact == 1 and nr == 1
    assert [c.source for c in kept] == ["a", "d"]


def test_html_extraction_strips_scripts(tmp_path):
    f = tmp_path / "p.html"
    f.write_text("<html><script>var x=1</script><nav>menu</nav><p>Hello &amp; welcome</p></html>")
    out = extract(f)
    assert "Hello & welcome" in out and "var x" not in out and "menu" not in out


def test_ingest_cli_writes_jsonl_and_survives_bad_file(tmp_path):
    (tmp_path / "a.txt").write_text(A + " " + A[:60] + ".\n")
    (tmp_path / "b.txt").write_text(B + "\n")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    out = tmp_path / "out.jsonl"
    res = CliRunner().invoke(app, ["data", "ingest", str(tmp_path), "--out", str(out), "--min-words", "5",
                                   "--json"])
    meta = json.loads(res.stdout)["meta"]
    assert meta["files_failed"] == 1 and meta["chunks_kept"] == 2
    rows = [json.loads(x) for x in out.read_text().splitlines() if x]
    assert {"text", "source", "index"} <= rows[0].keys()


def test_ingest_function_flags_unsupported(tmp_path):
    (tmp_path / "x.xyz").write_text("nope")
    _, meta, rep = ingest([tmp_path])
    assert meta["files_failed"] == 1 and any(d.code == "ETL002" for d in rep.items)
