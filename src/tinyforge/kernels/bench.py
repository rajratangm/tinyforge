"""`kernels verify` and `kernels bench`: prove kernels match PyTorch, then time real training runs."""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import torch

from . import registry
from .verify import verify

# SmolLM2-360M, the default test model of this project
DEFAULT_CFG = SimpleNamespace(hidden_size=960, intermediate_size=2560, num_attention_heads=15,
                              num_key_value_heads=5, vocab_size=49152)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
LOSS_TOL = 0.05  # relative; bf16 training over a few dozen steps drifts a little even with identical maths


def load_builtin_specs() -> None:
    """Import the built-in kernel modules (registering their specs); silently does nothing without Triton."""
    try:
        from . import add_rmsnorm, ce, rmsnorm, rope, swiglu  # noqa: F401
    except ImportError:
        pass


def shapes_for_config(cfg) -> dict[str, tuple[int, ...]]:
    heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    return {
        "rmsnorm": (2048, cfg.hidden_size),
        "swiglu": (2048, cfg.intermediate_size),
        "rope": (1, heads, getattr(cfg, "num_key_value_heads", None) or heads, 256, head_dim),
        "add_rmsnorm": (256, cfg.hidden_size),
        "fused_ce": (256, min(cfg.hidden_size, 512), cfg.vocab_size),
    }


def shapes_for_model(name: str) -> dict[str, tuple[int, ...]]:
    from transformers import AutoConfig

    return shapes_for_config(AutoConfig.from_pretrained(name))


def verify_rows(shapes: dict[str, tuple[int, ...]], dtype: torch.dtype, device: str = "cuda",
                only: str | None = None) -> list[dict]:
    rows = []
    for spec in registry.all_specs():
        if only and spec.name != only:
            continue
        shape = shapes.get(spec.name) or spec.example_shape
        row = {"kernel": spec.name, "shape": shape, "dtype": str(dtype).removeprefix("torch.")}
        if not shape:
            rows.append({**row, "ok": False, "reason": "no shape known for this kernel"})
        elif dtype not in spec.dtypes:
            rows.append({**row, "ok": False, "reason": "dtype not supported by this kernel"})
        else:
            tune = {k: v[0] for k, v in spec.tunables.items()}
            r = verify(spec, shape, dtype, device, **tune)
            rows.append({**row, "ok": r.ok, "max_err": r.max_err, "grad_err": r.grad_err, "reason": r.reason})
    return rows


def _close(a: list[float], b: list[float]) -> bool:
    n = min(len(a), len(b))
    return n > 0 and all(abs(x - y) <= LOSS_TOL * max(1.0, abs(x)) for x, y in zip(a[:n], b[:n], strict=True))


def _train_once(mode: str, model: str, steps: int, quant: str, data_dir: Path, workdir: Path,
                max_len: int) -> dict:
    run_dir = workdir / mode
    shutil.rmtree(run_dir, ignore_errors=True)  # our own scratch dir: always a fresh run, never a resume
    subprocess.run([sys.executable, "-m", "tinyforge", "ft", "train", "--base-model", model, "--steps",
                    str(steps), "--max-len", str(max_len), "--quant", quant, "--data-dir", str(data_dir),
                    "--run-dir", str(run_dir), "--kernels", mode], check=True)
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf8").splitlines()
    events = [json.loads(line) for line in lines]
    steps_ev = [e for e in events if e["event"] == "step"]
    finished = next(e for e in reversed(events) if e["event"] == "finished")
    kernels = next((e for e in events if e["event"] == "kernels"), {})
    # step 1 includes warm-up and kernel compilation, so the median skips it
    return {"tok_per_s": statistics.median(e["tok_per_s"] for e in steps_ev[1:] or steps_ev),
            "peak_mem_gb": finished["peak_mem_gb"], "losses": [e["loss"] for e in steps_ev],
            "final_loss": steps_ev[-1]["loss"],
            "kernel_report": {k: kernels[k] for k in ("patched", "skipped", "speedups") if k in kernels}}


def bench_model(model: str, steps: int = 30, quant: str = "none", data_dir: Path = Path("data/ft"),
                workdir: Path = Path("runs/kernel-bench"), max_len: int = 256,
                runner: Callable[[str], dict] | None = None) -> dict:
    """Train the same short run with kernels off and auto (separate processes, so peak memory is honest)."""
    runner = runner or (lambda mode: _train_once(mode, model, steps, quant, data_dir, workdir, max_len))
    off, auto = runner("off"), runner("auto")

    def summary(r: dict) -> dict:
        return {"tok_per_s": r["tok_per_s"], "peak_mem_gb": r["peak_mem_gb"], "final_loss": r["final_loss"]}

    return {"model": model, "steps": steps, "quant": quant, "off": summary(off), "auto": summary(auto),
            "loss_close": _close(off["losses"], auto["losses"]),
            "speedup": auto["tok_per_s"] / off["tok_per_s"] if off["tok_per_s"] else 0.0,
            "vram_saved_gb": off["peak_mem_gb"] - auto["peak_mem_gb"],
            "kernel_report": auto["kernel_report"]}
