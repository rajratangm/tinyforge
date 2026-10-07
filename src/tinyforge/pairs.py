"""Document chunks -> grounded question/answer training pairs, with the split made BEFORE generation.

Flow: chunks.jsonl (from `data ingest`) -> hash-split by source document into train/val -> a teacher model
proposes Q&A pairs per chunk -> a groundedness gate drops pairs whose answer is not supported by the chunk ->
chat JSONL.

The teacher is any OpenAI-compatible chat endpoint (llama.cpp, Ollama, vLLM, hosted APIs) or a callable.
Splitting by *source document* (not chunk) keeps neighbouring chunks of one file out of both sets, and
splitting before generation means teacher output can never leak a val question into train.

Each pair is reading comprehension (question + the passage in the prompt), not memorised facts.
Limits: the groundedness gate is lexical (answer content words must appear in the chunk), so it catches
invented facts but not subtle misreadings. Pairs inherit the teacher's errors and licence terms; `meta`
records the teacher and prompt for the data card.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROMPT = (
    "Read the passage and write {n} question-answer pairs a reader could answer using ONLY the passage. "
    "Answers must be short and copied or closely paraphrased from the passage. "
    'Reply with JSON only: [{{"question": "...", "answer": "..."}}].\n\nPassage:\n{chunk}'
)
STOP = set("a an the of to in and or is are was were be been it its for on at by with as that".split())

Teacher = Callable[[str], str]


def split_sources(rows: list[dict], val_pct: int = 10) -> tuple[list[dict], list[dict]]:
    """Deterministic split by source file; every chunk of a file lands on the same side."""
    def is_val(src: str) -> bool:
        return int(hashlib.sha256(src.encode()).hexdigest(), 16) % 100 < val_pct

    return [r for r in rows if not is_val(r["source"])], [r for r in rows if is_val(r["source"])]


def openai_teacher(base_url: str, model: str, api_key: str = "", timeout: float = 120.0) -> Teacher:
    import httpx

    def call(prompt: str) -> str:
        r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", timeout=timeout,
                       headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                       json={"model": model, "temperature": 0.2, "max_tokens": 600,
                             "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    return call


def _valid(i: object) -> bool:
    return isinstance(i, dict) and isinstance(i.get("question"), str) and isinstance(i.get("answer"), str)


def _parse(reply: str) -> list[dict]:
    """Q&A objects from a teacher reply. Tries the whole JSON array first; if the reply is fenced, has
    trailing prose, or was cut off mid-array, recovers each complete {...} object individually."""
    m = re.search(r"\[.*\]", reply, re.S)
    if m:
        try:
            items = json.loads(m.group(0))
            if isinstance(items, list):
                return [i for i in items if _valid(i)]
        except ValueError:
            pass
    found = []
    for obj in re.finditer(r"\{[^{}]*\}", reply):
        try:
            i = json.loads(obj.group(0))
        except ValueError:
            continue
        if _valid(i):
            found.append(i)
    return found


def _content_words(s: str) -> list[str]:
    return [w for w in re.findall(r"\w+", s.lower()) if w not in STOP]


def grounded(answer: str, chunk: str, min_frac: float = 0.8) -> bool:
    """Lexical support: at least min_frac of the answer's content words occur in the chunk."""
    words = _content_words(answer)
    if not words:
        return False
    have = set(re.findall(r"\w+", chunk.lower()))
    return sum(w in have for w in words) / len(words) >= min_frac


def generate(rows: list[dict], teacher: Teacher, per_chunk: int = 3,
             min_frac: float = 0.8) -> tuple[list[dict], dict]:
    out: list[dict] = []
    seen: set[str] = set()
    stats: dict[str, Any] = {
        "chunks": len(rows), "teacher_errors": 0, "unparseable": 0, "proposed": 0, "ungrounded": 0,
        "too_short": 0, "duplicate_question": 0, "kept": 0, "unparseable_samples": [],
    }
    for r in rows:
        try:
            reply = teacher(PROMPT.format(n=per_chunk, chunk=r["text"]))
        except Exception:  # a flaky teacher must not abort the batch
            stats["teacher_errors"] += 1
            continue
        items = _parse(reply)
        if not items:
            stats["unparseable"] += 1
            if len(stats["unparseable_samples"]) < 3:  # raw replies: what did the teacher return?
                stats["unparseable_samples"].append(reply[:300])
        for it in items[:per_chunk]:
            stats["proposed"] += 1
            q, a = it["question"].strip(), it["answer"].strip()
            if len(q.split()) < 3 or not a:
                stats["too_short"] += 1
            elif not grounded(a, r["text"], min_frac):
                stats["ungrounded"] += 1
            elif q.lower() in seen:
                stats["duplicate_question"] += 1
            else:
                seen.add(q.lower())
                out.append({"messages": [
                    {"role": "user", "content": f"{q}\n\nContext:\n{r['text']}"},
                    {"role": "assistant", "content": a}], "source": r["source"]})
                stats["kept"] += 1
    return out, stats


def run(chunks_path: Path, out_dir: Path, teacher: Teacher, teacher_name: str, per_chunk: int = 3,
        val_pct: int = 10, min_frac: float = 0.8) -> dict:
    rows = [json.loads(x) for x in chunks_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    train_rows, val_rows = split_sources(rows, val_pct)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"teacher": teacher_name, "prompt": PROMPT, "per_chunk": per_chunk,
                  "min_grounded_frac": min_frac,
                  "split": f"by source file, {val_pct}% val, done before generation",
                  "train_sources": len({r["source"] for r in train_rows}),
                  "val_sources": len({r["source"] for r in val_rows})}
    for name, part in (("train", train_rows), ("val", val_rows)):
        ex, stats = generate(part, teacher, per_chunk, min_frac)
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for e in ex:
                f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
        meta[name] = stats
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
