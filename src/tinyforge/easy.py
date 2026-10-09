"""`tinyforge easy`: turn "my files + a model name" into a fine-tuned model, with plain-English steps."""

from __future__ import annotations

import json
from pathlib import Path

CONTINUE = "Continue this text from my documents:\n\n"


def continuation_rows(chunks: list[dict], min_words: int = 40) -> list[dict]:
    """No teacher model: show the first half of a passage, ask for the second half.

    This teaches the wording and style of the documents. It does NOT teach answering questions."""
    rows = []
    for c in chunks:
        words = c["text"].split()
        if len(words) < min_words:
            continue
        mid = len(words) // 2
        rows.append({"messages": [
            {"role": "user", "content": CONTINUE + " ".join(words[:mid])},
            {"role": "assistant", "content": " ".join(words[mid:])}]})
    return rows


def build_training_file(data: Path, work: Path, teacher_url: str = "", teacher_model: str = "default",
                        api_key: str = "", per_chunk: int = 3) -> tuple[Path, str]:
    """Return (jsonl path in chat/instruction form, one plain sentence saying what was done)."""
    if not data.exists():
        raise FileNotFoundError(f"I cannot find '{data}'. Check the path (copy it from File Explorer).")
    if data.is_file() and data.suffix == ".jsonl":
        return data, "Your file already has examples, so I used it as it is."

    from . import etl, pairs

    chunks, _meta, _rep = etl.ingest([data])
    if not chunks:
        raise ValueError("I found no readable text (I read .txt, .md and .html; install tinyforge[docs] for "
                         "pdf/docx/pptx/xlsx).")
    work.mkdir(parents=True, exist_ok=True)
    chunk_file = work / "chunks.jsonl"
    rows = [{"text": c.text, "source": c.source, "index": c.index} for c in chunks]
    chunk_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    out = work / "examples.jsonl"

    if teacher_url:
        pdir = work / "pairs"
        pairs.run(chunk_file, pdir, pairs.openai_teacher(teacher_url, teacher_model, api_key),
                  f"{teacher_model}@{teacher_url}", per_chunk)
        lines = [ln for name in ("train", "val")
                 for ln in (pdir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out, f"A teacher model wrote {len(lines)} question-and-answer pairs from your documents."

    ex = continuation_rows(rows)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ex) + "\n", encoding="utf-8")
    return out, (f"Made {len(ex)} 'finish this text' examples. The model will learn the WORDING of your "
                 "documents, but not how to answer questions about them. For that, add --teacher-url "
                 "(any OpenAI-compatible chat endpoint).")
