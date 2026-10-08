"""Multi-GPU (data-parallel) LoRA fine-tuning on one machine.

`launch()` starts one process per GPU. Every process runs `finetune.train` (which detects WORLD_SIZE > 1 and
averages the LoRA gradients each step); only rank 0 reports events, which this module relays to the caller as
if training had run in-process.

Linux/macOS start the ranks with `torchrun` (elastic launcher). Windows PyTorch builds have no libuv, so
torchrun cannot create its rendezvous store there; on Windows the ranks are started directly and rendezvous
through a file (`TINYFORGE_DIST_INIT`). NCCL does not exist on Windows, so this path is for development and CI
(CPU, gloo), not for real multi-GPU runs.

Processes run `python -m tinyforge.ddp <config.json>`; callers use `launch`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from .finetune import FTConfig


def _start_ranks(cfg_path: Path, nproc: int, run_dir: Path) -> list[subprocess.Popen]:
    """Start the training processes. Element 0 is the one whose stdout carries the events."""
    env = dict(os.environ)
    if sys.platform != "win32":
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
               "-m", "tinyforge.ddp", str(cfg_path)]
        lead = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1, env=env)
        return [lead]
    init = (run_dir / "ddp_init").resolve()
    init.unlink(missing_ok=True)
    procs = []
    for rank in range(nproc):
        renv = {**env, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(nproc),
                "TINYFORGE_DIST_INIT": init.as_uri()}
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "tinyforge.ddp", str(cfg_path)], env=renv, text=True, bufsize=1,
            stdout=subprocess.PIPE if rank == 0 else subprocess.DEVNULL, stderr=sys.stderr))
    return procs


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


def launch(c: FTConfig, nproc: int, on_event: Callable[[dict], None] | None = None) -> dict:
    """Train with `nproc` processes (one per GPU); return the training summary like `finetune.train`."""
    emit = on_event or (lambda e: None)
    c.run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = c.run_dir / "ddp_config.json"
    cfg_path.write_text(c.model_dump_json(indent=2), encoding="utf-8")
    procs = _start_ranks(cfg_path, nproc, c.run_dir)

    def watch(p: subprocess.Popen) -> None:  # if any rank dies, the others would wait for it forever
        if p.wait() != 0:
            _stop(procs)

    for p in procs[1:]:
        threading.Thread(target=watch, args=(p,), daemon=True).start()
    summary: dict | None = None
    reason = ""
    try:
        assert procs[0].stdout is not None
        for line in procs[0].stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "finished":
                summary = {k: v for k, v in e.items() if k not in ("event", "t")}
            elif e.get("event") == "failed":
                reason = str(e.get("reason", ""))
            emit(e)
        code = procs[0].wait()
        for p in procs[1:]:
            code = code or p.wait()
    finally:
        _stop(procs)
    if code != 0 or summary is None:
        raise RuntimeError(reason or f"multi-GPU training failed (exit {code}); see the log above")
    return summary


def _main(cfg_file: str) -> None:
    from . import finetune

    c = FTConfig.model_validate_json(Path(cfg_file).read_text(encoding="utf-8"))

    def relay(e: dict) -> None:  # only rank 0 ever calls this
        print(json.dumps(e), flush=True)

    finetune.train(c, relay)


if __name__ == "__main__":
    _main(sys.argv[1])
