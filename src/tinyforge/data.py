"""Dataset preparation: fetch/ingest text, train BPE tokenizer, shard to uint16 binaries.

Includes data-quality checks that feed the warning system (tiny corpus, duplicates, etc.).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .warnings import Level, Report

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
SPECIALS = ["<pad>", "<bos>", "<eos>"]


def fetch_tinyshakespeare(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "raw.txt"
    if not out.exists():
        urllib.request.urlretrieve(SHAKESPEARE_URL, out)  # noqa: S310 (fixed https URL)
    return out


def train_tokenizer(text_path: Path, vocab_size: int, out_path: Path) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    tok.train([str(text_path)], trainer)
    tok.save(str(out_path))
    return tok


def load_tokenizer(data_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(data_dir) / "tokenizer.json"))


def quality_checks(text: str, n_tokens: int, vocab_size: int) -> Report:
    r = Report()
    if n_tokens < 100_000:
        r.add("DQ001", Level.WARN,
              f"Only {n_tokens:,} tokens: the model will memorise rather than generalise.",
              "Use more data or a smaller preset.")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        dup = 1 - len(set(lines)) / len(lines)
        if dup > 0.2:
            r.add("DQ002", Level.WARN, f"{dup:.0%} of lines are exact duplicates.",
                  "Deduplicate before training to avoid inflated eval scores.")
    ratio = n_tokens / max(1, len(text))
    if ratio > 0.6:
        r.add("DQ003", Level.WARN,
              f"Tokenizer averages {1/ratio:.1f} chars/token: vocab may be too small for this text.")
    if n_tokens / vocab_size < 50:
        r.add("DQ004", Level.WARN, "Fewer than 50 tokens per vocab entry; many tokens are undertrained.")
    return r


def prepare(text_path: Path, data_dir: Path, vocab_size: int = 2048,
            val_frac: float = 0.1) -> tuple[dict, Report]:
    """Tokenize and split. The split is contiguous (no leakage via shuffled overlapping windows)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    text = text_path.read_text(encoding="utf-8", errors="replace")
    tok = train_tokenizer(text_path, vocab_size, data_dir / "tokenizer.json")
    ids = np.asarray(tok.encode(text).ids, dtype=np.uint16)
    split = int(len(ids) * (1 - val_frac))
    ids[:split].tofile(data_dir / "train.bin")
    ids[split:].tofile(data_dir / "val.bin")
    meta = {
        "vocab_size": tok.get_vocab_size(), "train_tokens": int(split),
        "val_tokens": int(len(ids) - split),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source": str(text_path),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    rep = quality_checks(text, len(ids), meta["vocab_size"])
    if meta["val_tokens"] < 5_000:
        rep.add("DQ005", Level.ERROR, "Validation split under 5k tokens; eval will be noisy.")
    return meta, rep


class BinDataset:
    """Random-window sampler over a memory-mapped uint16 token file."""

    def __init__(self, path: Path, block_size: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block = block_size
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} has {len(self.data)} tokens, need > block_size+1")

    def batch(self, batch_size: int, device: str, rng: np.random.Generator):
        import torch

        ix = rng.integers(0, len(self.data) - self.block - 1, size=batch_size)
        x = np.stack([self.data[i:i + self.block] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.block] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
