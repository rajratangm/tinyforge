"""Fine-tuning data: ingest JSONL or a Hugging Face dataset -> normalised chat JSONL + quality gates."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator
from pathlib import Path

from . import pii as piilib
from .warnings import Level, Report

SECRET = piilib.SECRET_RE


def _iter_source(source: str, raw_limit: int | None) -> Iterator[dict]:
    p = Path(source)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            it = (json.loads(line) for line in f if line.strip())
            yield from itertools.islice(it, raw_limit)
        return
    from datasets import load_dataset

    ds = load_dataset(source, split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
    yield from itertools.islice(ds, raw_limit)


def normalize(r: dict) -> list[dict] | None:
    """Return a chat `messages` list ending in an assistant turn, or None if unusable."""
    if isinstance(r.get("messages"), list):
        msgs = [{"role": m.get("role"), "content": (m.get("content") or "").strip()} for m in r["messages"]]
        ok = msgs and msgs[-1]["role"] == "assistant" and any(m["role"] == "user" for m in msgs)
        return msgs if ok else None
    instr = r.get("instruction") or r.get("prompt") or r.get("question") or ""
    ctx = r.get("input") or r.get("context") or ""
    out = r.get("output") or r.get("response") or r.get("answer") or ""
    user = f"{instr}\n\n{ctx}".strip() if ctx else str(instr).strip()
    if not user:
        return None
    return [{"role": "user", "content": user}, {"role": "assistant", "content": str(out).strip()}]


def _key(msgs: list[dict]) -> str:
    return next(m["content"] for m in msgs if m["role"] == "user")


def _is_val(key: str, val_pct: int) -> bool:
    """Deterministic hash split: identical prompts can never land in both train and val."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100 < val_pct


def prepare(source: str, out_dir: Path, base_model: str, limit: int = 3000, val_pct: int = 5,
            max_len: int = 512, pii_policy: str = "flag") -> tuple[dict, Report]:
    """pii_policy: flag (keep, report), redact ([EMAIL] etc. placeholders) or drop (skip the example).
    Credential-like strings always drop the example."""
    if pii_policy not in ("flag", "redact", "drop"):
        raise ValueError("pii_policy must be flag, redact or drop")
    rep = Report()
    pii_kinds: dict[str, int] = {}
    pii_dropped = pii_redacted = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[list[dict]] = []
    raw = bad = dup = pii = 0
    for r in _iter_source(source, limit * 2):
        raw += 1
        msgs = normalize(r)
        if msgs is None or len(msgs[-1]["content"]) < 2:
            bad += 1
            continue
        k = _key(msgs)
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        found = piilib.scan(" ".join(m["content"] for m in msgs))
        if found:
            pii += 1
            for f in found:
                pii_kinds[f.kind] = pii_kinds.get(f.kind, 0) + 1
            if any(f.secret for f in found):
                continue  # never train on anything that looks like a credential
            if pii_policy == "drop":
                pii_dropped += 1
                continue
            if pii_policy == "redact":
                msgs = [{**m, "content": piilib.redact(m["content"])} for m in msgs]
                pii_redacted += 1
        rows.append(msgs)
        if len(rows) >= limit:
            break

    train = [m for m in rows if not _is_val(_key(m), val_pct)]
    val = [m for m in rows if _is_val(_key(m), val_pct)]
    for name, data in (("train", train), ("val", val)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for m in data:
                f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")

    # Length stats with the real tokenizer so truncation is predicted, not discovered.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    lens = [len(tok.apply_chat_template(m, tokenize=True, return_dict=False)) for m in train[:500]]
    trunc = sum(n > max_len for n in lens) / max(1, len(lens))

    meta = {"source": source, "base_model": base_model, "raw": raw, "train": len(train),
            "val": len(val), "dropped_invalid": bad, "dropped_duplicates": dup,
            "pii_flagged": pii, "pii_by_kind": pii_kinds, "pii_policy": pii_policy,
            "pii_dropped": pii_dropped, "pii_redacted": pii_redacted,
            "truncated_frac": trunc, "max_len": max_len}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if len(train) < 50:
        rep.add("FD001", Level.ERROR, f"Only {len(train)} training examples.",
                "Use a bigger source or raise --limit.")
    elif len(train) < 500:
        rep.add("FD001", Level.WARN, f"Only {len(train)} training examples; expect style transfer, "
                "not new knowledge.")
    if raw and dup / raw > 0.05:
        rep.add("FD002", Level.WARN, f"Removed {dup} duplicate prompts ({dup / raw:.0%} of input).")
    if raw and bad / raw > 0.1:
        rep.add("FD003", Level.WARN, f"Dropped {bad} malformed/empty examples ({bad / raw:.0%}).",
                "Check the dataset's column names.")
    if len(val) < 20:
        rep.add("FD004", Level.ERROR, f"Validation set has {len(val)} examples; eval would be noise.",
                "Raise --limit or --val-pct.")
    if trunc > 0.1:
        rep.add("FD005", Level.WARN, f"{trunc:.0%} of examples exceed {max_len} tokens and will be "
                "truncated.", "Raise --max-len (costs VRAM) or filter long examples.")
    if pii:
        rep.add("FD006", Level.WARN, f"{pii} examples contain PII or credential-like strings "
                f"({pii_kinds}; credential-like ones were dropped; policy={pii_policy}). Pattern-based: "
                "names and addresses are not detected.",
                "Use --pii redact or --pii drop before training anything you share.")
    return meta, rep
