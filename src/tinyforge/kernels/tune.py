"""Autotuner: pick the fastest verified configuration per (kernel, shape, dtype, GPU), and cache it."""

from __future__ import annotations

import itertools
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .registry import KernelSpec
from .verify import verify


@dataclass
class TuneResult:
    params: dict[str, int] = field(default_factory=dict)
    kernel_ms: float = 0.0
    ref_ms: float = 0.0
    adopted: bool = False
    reason: str = ""


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def _triton_version() -> str:
    try:
        import triton

        return triton.__version__
    except Exception:
        return "none"


def _cache_path(cache_dir: Path | None) -> Path:
    base = cache_dir or Path(os.environ.get("TINYFORGE_CACHE") or Path.home() / ".cache" / "tinyforge")
    return base / "kernel_tune.json"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return {}


def _store(path: Path, key: str, result: TuneResult) -> None:
    data = _load(path)
    data[key] = asdict(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf8")
    os.replace(tmp, path)


def _time(fn: Callable[[], object], label: str) -> float:
    """Median milliseconds over 10 runs after 3 warmups (CUDA events when on a GPU)."""
    for _ in range(3):
        fn()
    times = []
    if torch.cuda.is_available():
        for _ in range(10):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
    else:
        for _ in range(10):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def _runner(fn: Callable, base: tuple[torch.Tensor, ...], tune_kw: dict) -> Callable[[], None]:
    inputs = tuple(t.detach().clone().requires_grad_(t.is_floating_point()) for t in base)

    def run() -> None:
        out = fn(*inputs, **tune_kw)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        diff = [o for o in outs if o.requires_grad]
        if diff:
            torch.autograd.backward(diff, [torch.ones_like(o) for o in diff])
    return run


def tune(spec: KernelSpec, shape: tuple[int, ...], dtype: torch.dtype, device: str = "cuda",
         margin: float = 0.05, cache_dir: Path | None = None,
         time_fn: Callable[[Callable[[], object], str], float] | None = None) -> TuneResult:
    time_fn = time_fn or _time
    path = _cache_path(cache_dir)
    key = f"{spec.name}|{shape}|{dtype}|{_gpu_name()}|{_triton_version()}"
    hit = _load(path).get(key)
    if hit is not None:
        return TuneResult(**hit)

    names = list(spec.tunables)
    candidates = [dict(zip(names, vals, strict=True))
                  for vals in itertools.product(*(spec.tunables[n] for n in names))]
    first_failure = ""
    passing = []
    for params in candidates:
        v = verify(spec, shape, dtype, device, **params)
        if v.ok:
            passing.append(params)
        elif not first_failure:
            first_failure = v.reason
    if not passing:
        result = TuneResult(reason=f"no candidate passed verification: {first_failure}")
    else:
        torch.manual_seed(0)
        base = spec.make_inputs(shape, dtype, device)
        ref_ms = time_fn(_runner(spec.reference, base, {}), "ref")
        best, best_ms = passing[0], float("inf")
        for params in passing:
            label = ",".join(f"{k}={v}" for k, v in params.items()) or "default"
            ms = time_fn(_runner(spec.kernel, base, params), label)
            if ms < best_ms:
                best, best_ms = params, ms
        faster = best_ms <= ref_ms * (1 - margin)
        adopted = faster or spec.memory_kernel
        reason = "" if adopted else (f"not {margin:.0%} faster than the reference "
                                     f"({best_ms:.3f} vs {ref_ms:.3f} ms)")
        result = TuneResult(best, best_ms, ref_ms, adopted, reason)
    _store(path, key, result)
    return result
