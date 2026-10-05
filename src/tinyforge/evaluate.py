"""Evaluation suite with pass/warn/fail gates.

Checks: held-out perplexity (vs. uniform baseline), generation quality (repetition, distinct-n),
memorisation (n-gram copy rate against train set), determinism, throughput/latency, memory.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import hardware
from .data import BinDataset, load_tokenizer
from .infer import generate_ids, generate_text
from .model import GPT
from .train import estimate_loss, load_model
from .warnings import Level, Report

PROMPTS = ["ROMEO:", "To be, or not", "The king", "First Citizen:\n"]


def _distinct_n(ids: list[int], n: int) -> float:
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def _max_repeat_run(ids: list[int]) -> int:
    best = run = 1
    for a, b in zip(ids, ids[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _copy_rate(gen: list[int], train: np.ndarray, n: int = 8, cap: int = 2_000_000) -> float:
    """Fraction of generated n-grams appearing verbatim in (the first `cap` tokens of) train."""
    grams = {tuple(gen[i:i + n]) for i in range(len(gen) - n + 1)}
    if not grams:
        return 0.0
    t = train[:cap].tolist()
    seen = {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
    return len(grams & seen) / len(grams)


def evaluate(ckpt: Path, data_dir: Path, out_path: Path | None = None) -> dict:
    hw = hardware.probe()
    device = hw.device
    model, blob = load_model(ckpt, device)
    mc = model.c
    tok = load_tokenizer(data_dir)
    val = BinDataset(data_dir / "val.bin", mc.block_size)
    train_tokens = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
    rep = Report()
    res: dict = {"checkpoint": str(ckpt), "step": blob["step"], "params": model.num_params()}

    # 1. Perplexity
    amp = torch.bfloat16 if hw.bf16_supported and device == "cuda" else None
    vl = estimate_loss(model, val, 50, 8, device, amp)
    uniform = math.log(mc.vocab_size)
    res.update(val_loss=vl, val_ppl=math.exp(min(vl, 20)), uniform_baseline_loss=uniform)
    if vl >= uniform * 0.95:
        rep.add("EV001", Level.ERROR, f"Val loss {vl:.2f} ~ random guessing ({uniform:.2f}): "
                "model learned nothing.", "Train longer, check LR, verify data.")
    elif vl > uniform * 0.6:
        rep.add("EV002", Level.WARN, f"Val loss {vl:.2f} is weak vs. baseline {uniform:.2f}; "
                "model is undertrained.", "Increase --steps.")
    trl = estimate_loss(model, BinDataset(data_dir / "train.bin", mc.block_size), 50, 8, device, amp)
    res["train_loss"] = trl
    if vl - trl > 0.5:
        rep.add("EV003", Level.WARN, f"Generalisation gap {vl - trl:.2f} (val {vl:.2f} / train {trl:.2f}).",
                "Likely overfit: more data or smaller model.")

    # 2. Generation quality
    samples, d2, runs, copies = [], [], [], []
    for p in PROMPTS:
        ids = tok.encode(p).ids
        gen = list(generate_ids(model, ids, max_new=128, temperature=0.8, top_k=40, seed=0))
        samples.append({"prompt": p, "text": tok.decode(gen)})
        d2.append(_distinct_n(gen, 2))
        runs.append(_max_repeat_run(gen))
        copies.append(_copy_rate(gen, train_tokens))
    res["samples"] = samples
    res.update(distinct_2=float(np.mean(d2)), max_repeat_run=int(max(runs)),
               copy_rate_8gram=float(np.mean(copies)))
    if res["distinct_2"] < 0.5:
        rep.add("EV004", Level.WARN, f"Low diversity (distinct-2 = {res['distinct_2']:.2f}): "
                "degenerate/repetitive output.", "Raise temperature or train longer.")
    if res["max_repeat_run"] > 6:
        rep.add("EV005", Level.WARN, f"Token repeated {res['max_repeat_run']}x in a row.")
    if res["copy_rate_8gram"] > 0.6:
        rep.add("EV006", Level.WARN, f"{res['copy_rate_8gram']:.0%} of generated 8-grams are copied "
                "from the training set (memorisation).", "Use more data / regularisation.")

    # 3. Determinism (same seed => same output)
    a = generate_text(model, tok, "ROMEO:", max_new=40, seed=7)
    b = generate_text(model, tok, "ROMEO:", max_new=40, seed=7)
    res["deterministic"] = a == b
    if a != b:
        rep.add("EV007", Level.ERROR, "Seeded generation is not deterministic.")

    # 4. Cache correctness: KV-cache logits must match full forward pass
    ids = torch.tensor([tok.encode("To be, or not to be").ids], device=device)
    with torch.no_grad():
        full, _ = model(ids)
        caches = [{} for _ in model.blocks]
        model(ids[:, :-1], caches=caches, start_pos=0)
        inc, _ = model(ids[:, -1:], caches=caches, start_pos=ids.size(1) - 1)
    kv_err = (full - inc).abs().max().item()
    res["kv_cache_max_err"] = kv_err
    if kv_err > 1e-3:
        rep.add("EV008", Level.ERROR, f"KV-cache diverges from full forward (max err {kv_err:.2e}).")

    # 5. Throughput / latency
    res.update(_bench(model, device))
    if device == "cpu":
        rep.add("EV009", Level.INFO, "Benchmarks ran on CPU; GPU numbers will differ.")

    res["diagnostics"] = rep.to_list()
    res["passed"] = not rep.has_errors
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2))
    return res


@torch.no_grad()
def _bench(model: GPT, device: str, new_tokens: int = 64) -> dict:
    model.eval()
    prompt = [1] * 16
    list(generate_ids(model, prompt, max_new=4, temperature=0))  # warmup
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    n = sum(1 for _ in generate_ids(model, prompt, max_new=new_tokens, temperature=0))
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return {"decode_tok_per_s": n / dt if dt else 0.0, "ms_per_token": 1000 * dt / max(1, n),
            "inference_peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0}
