"""Optional training backend: Soup CLI (https://github.com/MakazhanAlpamys/Soup), exact layer streaming.

Why: layer streaming lets a 4 GB GPU fine-tune an 8B model (measured here on an RTX 3050 Ti Laptop:
Llama-3.1-8B, 51 steps at seq 512, ~176 tok/s, ~3.5 GB GPU in use). Soup is a separate, fast-moving,
single-maintainer BETA project, so it is kept at arm's length:

* never imported: it runs as a subprocess from its own virtualenv (`TINYFORGE_SOUP_BIN`, or `soup` on PATH);
* telemetry and the audit log are switched off (`--no-telemetry --no-audit-log`, `SOUP_TELEMETRY=0`) and the
  child runs with `HF_HUB_OFFLINE=1`: we download the base model first, then Soup never needs the network;
* only an allow-list of files is copied out of its output folder (adapter safetensors/config, tokenizer
  files); `training_args.bin` and other pickles are never loaded;
* its progress lines are translated into the worker contract's events (`started`, `step`, `finished`,
  `failed`).

Limits: preemption kills the child and loses progress since Soup's last checkpoint (no resume yet); there is
no per-step eval event, so `val_loss` gates fail closed on this backend; adapter loading is covered by tests
with a fake Soup plus the manual check recorded in benchmarks/.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

ALLOWED_OUT = ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "chat_template.jinja", "generation_config.json")
METRIC_DICT = re.compile(r"\{'(?:loss|train_runtime)'[^{}]*\}")  # may share a line with a progress bar


class BackendUnavailable(RuntimeError):
    """Soup is not installed (or not where we were told)."""


def find_soup() -> list[str]:
    explicit = os.environ.get("TINYFORGE_SOUP_BIN", "").strip()
    found = explicit or shutil.which("soup")
    if not found or not Path(found).exists() and not shutil.which(found):
        raise BackendUnavailable(
            "Soup is not installed. Install it in its OWN virtualenv (pip install 'soup-cli[train]') and set "
            "TINYFORGE_SOUP_BIN to that venv's soup executable.")
    return [found]


def ensure_local_model(base: str, cache_dir: Path, revision: str | None = None) -> Path:
    """Soup on Windows rejects the Hugging Face cache layout, so models must be plain local folders."""
    p = Path(base)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    target = cache_dir / re.sub(r"[^A-Za-z0-9._-]+", "__", base)
    snapshot_download(base, revision=revision, local_dir=str(target),
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja", "*.txt"],
                      ignore_patterns=["original/*"])
    return target


def build_config(doc: dict, model_dir: Path, train_path: Path, soup_out: Path) -> dict:
    """spec -> soup.yaml. Exact step control: the caller trims/cycles the data so one epoch == maxSteps."""
    s = doc["spec"]
    h, d = s.get("hyperparameters", {}), s["data"]
    return {
        "base": str(model_dir), "task": "sft",
        "data": {"train": str(train_path), "format": "chatml", "val_split": 0.0,
                 "max_length": d.get("maxLen", 512)},
        "training": {
            "epochs": 1, "lr": h.get("learningRate", 0.0002), "batch_size": h.get("batchSize", 1),
            "gradient_accumulation_steps": h.get("gradAccum", 1), "logging_steps": 5,
            "lora": {"r": h.get("loraR", 16), "alpha": h.get("loraAlpha", 32)},
            "quantization": "4bit" if s["method"] == "qlora" else "none",
            "stream_layers": True,
        },
        "output": str(soup_out),
    }


def make_train_file(src: Path, dst: Path, rows_needed: int) -> int:
    """Write exactly rows_needed rows to dst, cycling src if it is shorter. Returns distinct rows used."""
    rows = [x for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError(f"{src} has no training rows")
    with dst.open("w", encoding="utf-8") as f:
        for i in range(rows_needed):
            f.write(rows[i % len(rows)] + "\n")
    return min(len(rows), rows_needed)


def parse_metrics(line: str) -> dict | None:
    """Soup/Transformers log dicts look like {'loss': '0.12', 'epoch': '0.5', ...} with string values."""
    m = METRIC_DICT.search(line)
    if not m:
        return None
    try:
        raw = ast.literal_eval(m.group(0))
    except (ValueError, SyntaxError):
        return None
    out = {}
    for k, v in raw.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def collect_adapter(soup_out: Path, dest: Path) -> list[str]:
    """Copy allow-listed files into dest (the worker's `best/`). Raises if there is no adapter."""
    if not (soup_out / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Soup finished but wrote no adapter_model.safetensors in {soup_out}")
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ALLOWED_OUT:
        if (soup_out / name).exists():
            shutil.copy2(soup_out / name, dest / name)
            copied.append(name)
    return copied


def _lines(proc: subprocess.Popen):
    """Yield text lines, splitting on both \\n and \\r (Soup redraws progress bars with \\r)."""
    buf = b""
    stream = proc.stdout
    assert stream is not None  # run_soup opens the child with stdout=PIPE
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for p in parts:
            if p:
                yield p.decode("utf-8", "replace")
    if buf:
        yield buf.decode("utf-8", "replace")


def run_soup(doc: dict, out: Path, emit: Callable[..., None], model_dir: Path,
             soup_cmd: list[str] | None = None, popen=subprocess.Popen) -> tuple[int, str]:
    """Train with Soup. Returns (status, reason): 0 ok, 4 did not fit, 1 failure. Emits contract events."""
    out, model_dir = out.resolve(), model_dir.resolve()  # the child runs in another cwd: no relative paths
    s = doc["spec"]
    h = s.get("hyperparameters", {})
    steps = h.get("maxSteps", 300)
    eff =h.get("batchSize", 1) * h.get("gradAccum", 1)
    train_src = out / "data" / "train.jsonl"
    soup_out, workdir = out / "soup_out", out / "soup_work"
    workdir.mkdir(parents=True, exist_ok=True)
    distinct = make_train_file(train_src, workdir / "train.jsonl", steps * eff)
    cfg = build_config(doc, model_dir, workdir / "train.jsonl", soup_out)
    cfg_path = workdir / "soup.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cmd = (soup_cmd or find_soup()) + ["--no-telemetry", "--no-audit-log", "train", "--config", str(cfg_path),
                                       "--yes"]
    env = {**os.environ, "SOUP_TELEMETRY": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "WANDB_DISABLED": "true", "PYTHONUTF8": "1",
           "PYTHONUNBUFFERED": "1"}  # piped stdout is block-buffered otherwise: no live progress
    emit("started", backend="soup", steps=steps, examples_per_step=eff, distinct_examples=distinct,
         model=str(model_dir), stream_layers=True)
    t0 = last_t = time.time()
    last_tokens = 0.0
    last_loss = None
    tail: list[str] = []
    proc = popen(cmd, cwd=str(workdir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for line in _lines(proc):
            tail = (tail + [line])[-40:]
            m = parse_metrics(line)
            if not m or "loss" not in m:
                continue
            now = time.time()
            tokens = m.get("num_tokens", last_tokens)
            step = min(steps, max(1, int(m.get("epoch", 0.0) * steps + 0.5)))  # half-up, not banker's
            last_loss = m["loss"]
            dt = now - last_t
            rate = (tokens - last_tokens) / dt if tokens >= last_tokens and dt >= 0.05 else None
            emit("step", step=step, loss=m["loss"], lr=m.get("learning_rate"), grad_norm=m.get("grad_norm"),
                 tok_per_s=rate, peak_mem_gb=None)
            last_t, last_tokens = now, tokens
        code = proc.wait()
    except BaseException:  # preemption or Ctrl-C: do not leave a GPU-holding orphan
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            proc.kill()
        raise
    blob = "\n".join(tail).lower()
    if code != 0:
        oom = "out of memory" in blob or "outofmemory" in blob
        reason = "out of memory" if oom else f"soup exited {code}"
        emit("failed", reason=reason, tail=tail[-8:])
        return (4 if reason == "out of memory" else 1), reason
    if last_loss is None or (isinstance(last_loss, float) and math.isnan(last_loss)):
        emit("failed", reason="soup produced no finite loss", tail=tail[-8:])
        return 1, "no loss"
    try:
        files = collect_adapter(soup_out, out / "best")
    except FileNotFoundError as exc:
        emit("failed", reason=str(exc), tail=tail[-8:])
        return 1, "no adapter"
    digest = hashlib.sha256((out / "best" / "adapter_model.safetensors").read_bytes()).hexdigest()
    emit("artifact", kind="adapter", path=str(out / "best"), files=files, sha256=digest)
    emit("finished", steps=steps, final_train_loss=last_loss, backend="soup",
         seconds=round(time.time() - t0, 1), peak_mem_gb=None)
    return 0, ""


def write_run_config(out: Path, model_dir: Path, doc: dict) -> None:
    """ft_config.json so `ft eval`, the serving backend and the model card can load the adapter."""
    s = doc["spec"]
    h = s.get("hyperparameters", {})
    from .finetune import FTConfig

    cfg = FTConfig(base_model=str(model_dir), run_dir=out, data_dir=out / "data",
                   quant="4bit" if s["method"] == "qlora" else "none", max_len=s["data"].get("maxLen", 512),
                   max_steps=h.get("maxSteps", 300), batch_size=h.get("batchSize", 1),
                   grad_accum=h.get("gradAccum", 1), lr=h.get("learningRate", 0.0002),
                   lora_r=h.get("loraR", 16), lora_alpha=h.get("loraAlpha", 32), auto_plan=False)
    (out / "ft_config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8")
