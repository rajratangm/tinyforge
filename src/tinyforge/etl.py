"""Document ETL: files -> clean text chunks (JSONL) with near-duplicate removal.

Stages: extract -> clean -> chunk -> dedupe. Plain text, Markdown and HTML use the standard library;
PDF/DOCX/PPTX/XLSX need the optional `markitdown` package (`pip install markitdown`). Making training pairs
from the chunks is a separate, verified step (not here): split train/val BEFORE generating pairs so teacher
output cannot leak eval questions.

Dedupe is exact-hash plus MinHash/LSH over word 5-shingles, fine for up to ~100k chunks in memory; use
datatrove/text-dedup beyond that.
"""
from __future__ import annotations

import hashlib
import html
import re
import struct
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from . import pii as piilib
from .warnings import Level, Report

TEXT_EXT = {".txt", ".md", ".markdown", ".rst"}
HTML_EXT = {".html", ".htm"}
MARKITDOWN_EXT = {".pdf", ".docx", ".pptx", ".xlsx"}
NUM_PERM = 64
BANDS = 16                  # 16 bands x 4 rows: catches pairs well above ~0.6 Jaccard
SHINGLE = 5
_MASK = (1 << 61) - 1


@dataclass
class Chunk:
    text: str
    source: str
    index: int


class _Strip(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def extract(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext in HTML_EXT:
        p = _Strip()
        p.feed(path.read_text(encoding="utf-8", errors="replace"))
        return html.unescape("".join(p.out))
    if ext in MARKITDOWN_EXT:
        try:
            from markitdown import MarkItDown
        except ImportError as e:
            raise RuntimeError(f"{path.name}: reading {ext} needs the 'docs' extra (markitdown)") from e
        return MarkItDown().convert(str(path)).text_content
    raise RuntimeError(f"{path.name}: unsupported file type {ext!r}")


_STRUCTURAL = ("|", "#", "-", "*", ">", "`", "+")
_ENDS_SENTENCE = (".", "!", "?", ":", ";", '"', "'", ")", "]", "}", "`")


def _is_boilerplate_candidate(ln: str) -> bool:
    """Short, label-like line: no sentence punctuation, not a table/list/heading/code line, has a letter."""
    return (0 < len(ln) < 80 and not ln.startswith(_STRUCTURAL) and not ln.endswith(_ENDS_SENTENCE)
            and any(c.isalpha() for c in ln))


def _rejoin_wrapped(text: str) -> str:
    """PDF extractors often emit a blank line at every visual line wrap; glue those fragments back."""
    out: list[str] = []
    for block in text.split("\n\n"):
        prev = out[-1] if out else ""
        open_ended = prev and not prev.startswith(_STRUCTURAL) and not prev.endswith(_ENDS_SENTENCE)
        if open_ended and "\n" not in prev and block[:1].islower():
            out[-1] = prev + " " + block
        else:
            out.append(block)
    return "\n\n".join(out)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join words hyphenated across lines
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    # Page headers/footers/menus repeat 3+ times (page numbers normalised so "Page 1"/"Page 2" match). Only
    # label-like lines qualify: repeated sentences, table rows, list items and code lines are content.
    in_code, kept = False, []
    norm = [re.sub(r"\d+", "#", ln.lower()) for ln in lines]
    freq = Counter(n for ln, n in zip(lines, norm, strict=True) if _is_boilerplate_candidate(ln))
    for ln, n in zip(lines, norm, strict=True):
        if ln.startswith("```"):
            in_code = not in_code
        if not in_code and _is_boilerplate_candidate(ln) and freq[n] >= 3:
            continue
        kept.append(ln)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return _rejoin_wrapped(text)


def chunk_text(text: str, max_words: int = 300) -> list[str]:
    """Paragraph-aware chunks of at most ~max_words words (a single huge paragraph is split by words)."""
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        words = para.split()
        if len(words) > max_words:
            if cur:
                out.append("\n\n".join(cur))
                cur, n = [], 0
            out.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
            continue
        if n + len(words) > max_words and cur:
            out.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(para)
        n += len(words)
    if cur:
        out.append("\n\n".join(cur))
    return out


# ------------------------------------------------------------------ dedupe

def _h64(s: str) -> int:
    return struct.unpack("<Q", hashlib.blake2b(s.encode(), digest_size=8).digest())[0]


def _signature(text: str) -> tuple[int, ...] | None:
    w = re.findall(r"\w+", text.lower())
    if len(w) < SHINGLE:
        return None
    base = {_h64(" ".join(w[i:i + SHINGLE])) for i in range(len(w) - SHINGLE + 1)}
    return tuple(min(((h * (2 * k + 1) + k * 0x9E3779B97F4A7C15) & _MASK) for h in base)
                 for k in range(NUM_PERM))


def _jaccard_est(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def dedupe(chunks: list[Chunk], threshold: float = 0.8) -> tuple[list[Chunk], int, int]:
    """Drop exact duplicates, then near-duplicates (estimated Jaccard >= threshold). Keeps the first seen.

    Returns (kept, exact_dropped, near_dropped).
    """
    kept: list[Chunk] = []
    sigs: list[tuple[int, ...] | None] = []
    seen: set[str] = set()
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    rows = NUM_PERM // BANDS
    exact = near = 0
    for c in chunks:
        key = hashlib.sha256(" ".join(c.text.split()).lower().encode()).hexdigest()
        if key in seen:
            exact += 1
            continue
        seen.add(key)
        sig = _signature(c.text)
        if sig is not None:
            cands = {i for b in range(BANDS)
                     for i in buckets.get((b, repr(sig[b * rows:(b + 1) * rows]).encode()), ())}
            if any(sigs[i] is not None and _jaccard_est(sig, sigs[i]) >= threshold for i in cands):
                near += 1
                continue
            for b in range(BANDS):
                buckets[(b, repr(sig[b * rows:(b + 1) * rows]).encode())].append(len(kept))
        sigs.append(sig)
        kept.append(c)
    return kept, exact, near


# ------------------------------------------------------------------ pipeline

def ingest(paths: list[Path], max_words: int = 300, min_words: int = 20, threshold: float = 0.8,
           pii_policy: str = "flag") -> tuple[list[Chunk], dict, Report]:
    """pii_policy: flag | redact | drop. Chunks containing credentials are always dropped."""
    if pii_policy not in ("flag", "redact", "drop"):
        raise ValueError("pii_policy must be flag, redact or drop")
    rep = Report()
    pii_kinds: dict[str, int] = {}
    secret_dropped = pii_dropped = 0
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p])
    chunks: list[Chunk] = []
    failed = short = 0
    for f in files:
        try:
            text = clean(extract(f))
        except Exception as e:  # one bad file must not sink the batch
            failed += 1
            rep.add("ETL001", Level.WARN, str(e))
            continue
        for i, c in enumerate(chunk_text(text, max_words)):
            if len(c.split()) < min_words:
                short += 1
                continue
            found = piilib.scan(c)
            for x in found:
                pii_kinds[x.kind] = pii_kinds.get(x.kind, 0) + 1
            if any(x.secret for x in found):
                secret_dropped += 1
                continue
            if found and pii_policy == "drop":
                pii_dropped += 1
                continue
            if found and pii_policy == "redact":
                c = piilib.redact(c, found)
            chunks.append(Chunk(c, str(f), i))
    kept, exact, near = dedupe(chunks, threshold)
    meta = {"files": len(files), "files_failed": failed, "chunks_made": len(chunks), "dropped_short": short,
            "dropped_exact_dup": exact, "dropped_near_dup": near, "chunks_kept": len(kept),
            "pii_by_kind": pii_kinds, "pii_policy": pii_policy, "dropped_secret": secret_dropped,
            "dropped_pii": pii_dropped}
    if pii_kinds:
        rep.add("ETL005", Level.WARN, f"PII or secrets found in chunks: {pii_kinds} (policy={pii_policy}).",
                "Use --pii redact or --pii drop; names and addresses are not detected.")
    if not kept:
        rep.add("ETL002", Level.ERROR, "No usable text came out.",
                "Check file types and that files are not scans.")
    elif chunks and (exact + near) / len(chunks) > 0.3:
        rep.add("ETL003", Level.WARN, f"{(exact + near) / len(chunks):.0%} of chunks were duplicates.",
                "Check for repeated pages or overlapping exports.")
    if kept and sum(len(c.text.split()) for c in kept) < 2000:
        rep.add("ETL004", Level.WARN, "Under 2,000 words in total: too little to teach a model anything new.")
    return kept, meta, rep
