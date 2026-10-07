"""Memory-ladder benchmark: short real training runs, one fresh process per variant, raw JSON out.

  python tools/bench_memory.py --model Qwen/Qwen2.5-1.5B-Instruct --data-dir data/sql --steps 15 \
      --variant base:chunked_ce=false --variant chunked:chunked_ce=true --out benchmarks/chunked-ce-1p5b.json

A variant is name:key=value,key=value over FTConfig fields. Records peak allocated VRAM, median tok/s over the
logged steps, final train loss and success or the failure reason. Methodology: first logged step excluded from
tok/s (warm-up), identical seed/data/steps across variants, one process each so peaks do not leak.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def worker(cfg_json: str) -> None:
    from tinyforge.finetune import FTConfig, train

    c = FTConfig.model_validate_json(cfg_json)
    events: list[dict] = []
    train(c, on_event=events.append)
    steps = [e for e in events if e.get("kind") == "step" or e.get("event") == "step"]
    print("RESULT " + json.dumps({"events": steps, "n_events": len(events)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="data/sql")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return

    import torch
    out = {"model": a.model, "steps": a.steps, "max_len": a.max_len, "data_dir": a.data_dir,
           "env": {"python": platform.python_version(), "torch": torch.__version__, "os": platform.platform(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                   if torch.cuda.is_available() else 0},
           "variants": []}
    for spec in a.variant:
        name, _, kv = spec.partition(":")
        over = {k: _coerce(v) for k, v in (p.split("=", 1) for p in kv.split(",") if p)}
        cfg = {"base_model": a.model, "data_dir": a.data_dir, "max_steps": a.steps, "max_len": a.max_len,
               "batch_size": a.batch_size, "grad_accum": a.grad_accum, "eval_interval": 10_000,
               "eval_examples": 16, "auto_plan": False, "run_dir": f"runs/bench-{name}", **over}
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, "--model", a.model, "--variant", "x", "--out", "x",
                            "--worker", json.dumps(cfg)], capture_output=True, text=True)
        rec = {"name": name, "overrides": over, "wall_s": round(time.time() - t0, 1)}
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if p.returncode == 0 and line:
            ev = json.loads(line[7:])["events"]
            tps = [e["tok_per_s"] for e in ev[1:] if e.get("tok_per_s")]
            rec.update(ok=True, peak_mem_gb=max((e.get("peak_mem_gb", 0) for e in ev), default=0),
                       median_tok_per_s=statistics.median(tps) if tps else None,
                       last_loss=ev[-1]["loss"] if ev else None, step_events=len(ev))
        else:
            tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
            rec.update(ok=False, failure=" | ".join(tail)[:400])
        print(json.dumps(rec))
        out["variants"].append(rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
