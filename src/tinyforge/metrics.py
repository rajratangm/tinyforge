"""Prometheus text exposition (format 0.0.4) for the API. Hand-written: no client library dependency.

Cardinality is bounded on purpose: the only labels are the fixed job status set, histogram `le`, and the
version string. Job ids never appear as labels. Training signals are read from the JSON-lines events the
worker already prints (`step`, `eval`) and are emitted only while a job is running, never as stale values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

STATUSES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
DURATION_STATUSES = ("done", "failed")  # cancelled/interrupted durations are not meaningful
DURATION_BUCKETS = (10.0, 30.0, 60.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0, 21600.0)
LOG_SCAN_LINES = 500  # how far back in the running job's log to look for the latest step/eval


def _esc(value: str) -> str:
    """Escape a label value per the exposition format: backslash, double quote, newline."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _num(v: float) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    return repr(float(v))


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def latest_train_signals(log_lines: Iterable[str]) -> dict[str, float]:
    """Newest step/eval values from a running job's log, or {} if there is nothing current.

    Scans backwards. A `finished` or `failed` event newer than the last step means training is over (the job
    may be in a later stage such as eval), so no training signals are reported.
    """
    out: dict[str, float] = {}
    seen_step = seen_eval = False
    for line in reversed(list(log_lines)[-LOG_SCAN_LINES:]):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        kind = e.get("event") if isinstance(e, dict) else None
        if kind in ("finished", "failed") and not seen_step:
            return {}
        if kind == "step" and not seen_step:
            seen_step = True
            for src, name in (("step", "step"), ("loss", "loss"), ("tok_per_s", "tok_per_second"),
                              ("peak_mem_gb", "peak_mem_gb")):
                if _is_number(e.get(src)):
                    out[name] = e[src]
        elif kind == "eval" and not seen_eval:
            seen_eval = True
            if _is_number(e.get("val_loss")):
                out["val_loss"] = e["val_loss"]
        if seen_step and seen_eval:
            break
    return out


def render(version: str, jobs: list[dict], running_log: Iterable[str] = (),
           host: dict[str, float] | None = None, guard: dict[str, int] | None = None) -> str:
    """Build the /metrics body from job rows (JobStore.list) and the running job's in-memory log lines.

    host: optional memory gauges in bytes (ram_total, ram_available, swap_used). Swap or a collapsing
    ram_available during offloaded training is the early warning for a 10x slowdown."""
    counts = dict.fromkeys(STATUSES, 0)
    for j in jobs:
        if j["status"] in counts:
            counts[j["status"]] += 1

    o: list[str] = []

    def metric(name: str, kind: str, help_: str) -> None:
        o.append(f"# HELP {name} {help_}")
        o.append(f"# TYPE {name} {kind}")

    metric("tinyforge_build_info", "gauge", "Build information; the value is always 1.")
    o.append(f'tinyforge_build_info{{version="{_esc(version)}"}} 1')

    metric("tinyforge_jobs", "gauge", "Jobs in the queue database by status.")
    for s in STATUSES:
        o.append(f'tinyforge_jobs{{status="{s}"}} {counts[s]}')

    metric("tinyforge_queue_depth", "gauge", "Jobs waiting to run.")
    o.append(f"tinyforge_queue_depth {counts['queued']}")

    metric("tinyforge_job_duration_seconds", "histogram",
           "Wall-clock duration of finished jobs (done and failed only).")
    for s in DURATION_STATUSES:
        durs = [j["finished"] - j["started"] for j in jobs
                if j["status"] == s and j.get("started") is not None and j.get("finished") is not None]
        for le in DURATION_BUCKETS:
            n = sum(1 for d in durs if d <= le)
            o.append(f'tinyforge_job_duration_seconds_bucket{{status="{s}",le="{_num(le)}"}} {n}')
        o.append(f'tinyforge_job_duration_seconds_bucket{{status="{s}",le="+Inf"}} {len(durs)}')
        o.append(f'tinyforge_job_duration_seconds_sum{{status="{s}"}} {_num(sum(durs))}')
        o.append(f'tinyforge_job_duration_seconds_count{{status="{s}"}} {len(durs)}')

    sig = latest_train_signals(running_log) if counts["running"] else {}
    for key, help_ in (("step", "Latest optimizer step of the running job."),
                       ("loss", "Latest training loss of the running job."),
                       ("tok_per_second", "Latest training throughput (tokens/s) of the running job."),
                       ("peak_mem_gb", "Peak GPU memory (GB) reported by the running job."),
                       ("val_loss", "Latest validation loss of the running job.")):
        if key in sig:
            name = f"tinyforge_{'val_loss' if key == 'val_loss' else 'train_' + key}"
            metric(name, "gauge", help_)
            o.append(f"{name} {_num(sig[key])}")

    for key, name, help_ in (("ram_total", "tinyforge_host_ram_total_bytes", "Host RAM installed."),
                             ("ram_available", "tinyforge_host_ram_available_bytes", "Host RAM available."),
                             ("swap_used", "tinyforge_host_swap_used_bytes", "Host swap/pagefile in use.")):
        if host and _is_number(host.get(key)):
            metric(name, "gauge", help_)
            o.append(f"{name} {_num(host[key])}")

    if guard:
        metric("tinyforge_guard_events_total", "counter", "Guardrail actions (blocked/redacted) by reason.")
        for reason in sorted(guard):
            o.append(f'tinyforge_guard_events_total{{reason="{_esc(reason)}"}} {int(guard[reason])}')

    return "\n".join(o) + "\n"
