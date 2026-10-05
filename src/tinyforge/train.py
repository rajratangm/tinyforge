"""Training loop: AMP, grad accumulation, checkpointing, resume, cosine LR, health guards.

All progress is emitted as JSON-lines events (metrics.jsonl + callback) so the CLI, API
and UI share one source of truth.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from . import hardware
from .config import ModelConfig, TrainConfig, plan
from .data import BinDataset
from .model import GPT

Event = dict
Callback = Callable[[Event], None]


def lr_at(step: int, c: TrainConfig) -> float:
    if step < c.warmup_steps:
        return c.lr * (step + 1) / c.warmup_steps
    prog = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return c.lr * (c.min_lr_frac + (1 - c.min_lr_frac) * cos)


def save_checkpoint(path: Path, model: GPT, opt, step: int, mc: ModelConfig, tc: TrainConfig,
                    best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None,
                "step": step, "model_config": mc.model_dump(),
                "train_config": tc.model_dump(mode="json"), "best_val": best_val}, tmp)
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint


def load_model(ckpt: Path, device: str) -> tuple[GPT, dict]:
    blob = torch.load(ckpt, map_location=device, weights_only=True)  # never unpickle arbitrary objects
    model = GPT(ModelConfig(**blob["model_config"])).to(device)
    model.load_state_dict(blob["model"])
    return model, blob


@torch.no_grad()
def estimate_loss(model: GPT, ds: BinDataset, batches: int, bs: int, device: str,
                  amp_dtype, seed: int = 0) -> float:
    model.eval()
    rng = np.random.default_rng(seed)  # fixed seed => comparable across checkpoints
    total = 0.0
    for _ in range(batches):
        x, y = ds.batch(bs, device, rng)
        with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / batches


def train(mc: ModelConfig, tc: TrainConfig, on_event: Callback | None = None) -> dict:
    emit = on_event or (lambda e: None)
    tc.run_dir.mkdir(parents=True, exist_ok=True)
    log = open(tc.run_dir / "metrics.jsonl", "a", buffering=1)

    def event(kind: str, **kw):
        e = {"event": kind, "t": time.time(), **kw}
        log.write(json.dumps(e) + "\n")
        emit(e)

    hw = hardware.probe()
    tc, rep = plan(mc, tc, hw.vram_gb, hw.bf16_supported)
    for d in rep.to_list():
        event("diagnostic", **d)
    if rep.has_errors:
        event("failed", reason="plan did not fit hardware")
        raise RuntimeError("Training plan does not fit available hardware; see diagnostics.")

    device = hw.device
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[tc.precision]
    if device == "cpu":
        amp_dtype = None
    train_ds = BinDataset(tc.data_dir / "train.bin", mc.block_size)
    val_ds = BinDataset(tc.data_dir / "val.bin", mc.block_size)

    model = GPT(mc).to(device)
    model.grad_checkpointing = tc.grad_checkpointing
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tc.lr, betas=(0.9, 0.95), fused=(device == "cuda"))
    scaler = torch.amp.GradScaler(device, enabled=amp_dtype is torch.float16)

    step, best_val = 0, float("inf")
    last = tc.run_dir / "last.pt"
    if last.exists():
        blob = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        step, best_val = blob["step"], blob["best_val"]
        event("resumed", step=step)

    run_model = torch.compile(model) if tc.compile else model
    event("started", params=model.num_params(), device=device, precision=tc.precision,
          max_steps=tc.max_steps, config=tc.model_dump(mode="json"))

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0, tok_count = time.time(), 0
    val_history: list[float] = []
    bad_streak = 0
    while step < tc.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        loss_acc = 0.0
        for _ in range(tc.grad_accum):
            x, y = train_ds.batch(tc.batch_size, device, rng)
            with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = run_model(x, y)
            scaler.scale(loss / tc.grad_accum).backward()
            loss_acc += loss.item() / tc.grad_accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        if not math.isfinite(loss_acc) or not math.isfinite(gnorm):
            bad_streak += 1
            opt.zero_grad(set_to_none=True)
            event("diagnostic", code="TR001", level="warn", fix="",
                  message=f"Non-finite loss/grad at step {step}; step skipped ({bad_streak}/5).")
            if bad_streak >= 5:
                event("failed", reason="training diverged (NaN/inf)")
                raise RuntimeError("Training diverged. Lower --lr or use --precision fp32.")
            scaler.update()
            step += 1
            continue
        bad_streak = 0
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_count += tc.batch_size * tc.grad_accum * mc.block_size

        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
            event("step", step=step, loss=loss_acc, lr=lr_at(step, tc), grad_norm=gnorm,
                  tok_per_s=tok_count / dt if dt > 0 else 0.0, peak_mem_gb=mem)
            t0, tok_count = time.time(), 0

        if step % tc.eval_interval == 0 or step == tc.max_steps:
            amp = amp_dtype
            vl = estimate_loss(model, val_ds, tc.eval_batches, tc.batch_size, device, amp)
            tl_ = estimate_loss(model, train_ds, tc.eval_batches, tc.batch_size, device, amp)
            val_history.append(vl)
            event("eval", step=step, val_loss=vl, train_loss=tl_, val_ppl=math.exp(min(vl, 20)))
            if vl < best_val:
                best_val = vl
                save_checkpoint(tc.run_dir / "best.pt", model, None, step, mc, tc, best_val)
            if vl - tl_ > 0.5 and step >= 2 * tc.eval_interval:
                event("diagnostic", code="TR002", level="warn",
                      message=f"Overfitting: val {vl:.2f} vs train {tl_:.2f}.",
                      fix="More data, dropout, smaller preset, or stop earlier (best.pt is kept).")
            if len(val_history) >= 4 and min(val_history[-3:]) > min(val_history[:-3]):
                event("diagnostic", code="TR003", level="warn", fix="",
                      message="Validation loss has not improved for 3 evals (plateau).")

        if step % tc.ckpt_interval == 0 or step == tc.max_steps:
            save_checkpoint(last, model, opt, step, mc, tc, best_val)

    summary = {"steps": step, "best_val_loss": best_val,
               "best_val_ppl": math.exp(min(best_val, 20)),
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0}
    event("finished", **summary)
    log.close()
    return summary
