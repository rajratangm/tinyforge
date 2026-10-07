"""Serving telemetry for the dashboard: memory over time, KV-cache math, and one record per request.

KV cache size is exact arithmetic, not a guess: every token stores a key and a value vector in every layer, so
    bytes_per_token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_value
and it grows linearly with the tokens held in context. `kv_curve` returns that line for charting.

What the numbers mean (they differ by engine, and the dashboard says which):
* llama.cpp allocates the whole KV buffer up front for `context x slots` tokens, so measured VRAM stays flat while
  the *used* part of that buffer grows with the tokens in context.
* The in-process Hugging Face backend grows its cache token by token, so measured VRAM does rise with context.

Sampling is a daemon thread (default every 2 s) reading nvidia-smi and psutil; requests are recorded by the router
callback. Everything is in memory (ring buffers): a restart clears it. Honest limits: GPU numbers are whole-device
(other programs count), the per-request KV figure is tokens x bytes/token (an estimate of use, not a probe of the
engine's buffer), and without an NVIDIA GPU the GPU series is empty.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import psutil

GB = 1024**3
MB = 1024**2
CACHE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}


def kv_bytes_per_token(
    layers: int, kv_heads: int, head_dim: int, k_type: str = "f16", v_type: str = "f16"
) -> float:
    """Exact KV bytes one token adds: K and V vectors in every layer (types may differ, e.g. q8_0 keys)."""
    kb, vb = CACHE_BYTES.get(k_type, 2.0), CACHE_BYTES.get(v_type, 2.0)
    return layers * kv_heads * head_dim * (kb + vb)


def kv_info_from_hf_config(cfg: dict, k_type: str = "f16", v_type: str = "f16") -> dict | None:
    try:
        layers = int(cfg["num_hidden_layers"])
        heads = int(cfg["num_attention_heads"])
        kv_heads = int(cfg.get("num_key_value_heads") or heads)
        head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // heads)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "bytes_per_token": kv_bytes_per_token(layers, kv_heads, head_dim, k_type, v_type),
        "k_type": k_type,
        "v_type": v_type,
        "source": "model config",
    }


def kv_info_from_gguf(
    path: Path, gguf_py: Path | None = None, k_type: str = "f16", v_type: str = "f16"
) -> dict | None:
    """Read layer/head numbers from a GGUF header (needs the gguf python package; None if unavailable)."""
    import sys

    try:
        if gguf_py and str(gguf_py) not in sys.path:
            sys.path.insert(0, str(gguf_py))
        from gguf import GGUFReader

        r = GGUFReader(str(path))
        arch = str(r.fields["general.architecture"].contents())

        def get(key: str):
            f = r.fields.get(f"{arch}.{key}")
            return None if f is None else f.contents()

        layers, heads = int(get("block_count")), int(get("attention.head_count"))
        kv_heads = int(get("attention.head_count_kv") or heads)
        head_dim = int(get("attention.key_length") or int(get("embedding_length")) // heads)
    except Exception:  # noqa: BLE001 - optional enrichment must never break serving
        return None
    return {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "bytes_per_token": kv_bytes_per_token(layers, kv_heads, head_dim, k_type, v_type),
        "k_type": k_type,
        "v_type": v_type,
        "source": "GGUF header",
    }


def kv_curve(bytes_per_token: float, max_tokens: int, points: int = 32) -> list[dict]:
    step = max(1, max_tokens // points)
    return [{"tokens": t, "mb": round(t * bytes_per_token / MB, 2)} for t in range(0, max_tokens + 1, step)]


def gpu_memory() -> tuple[float, float] | None:
    """(used_gb, total_gb) of GPU 0 via nvidia-smi, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = (
            subprocess.run(
                [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        used, total = (float(x) for x in out.split(","))
        return used / 1024, total / 1024
    except Exception:  # noqa: BLE001
        return None


class Telemetry:
    def __init__(self, interval_s: float = 2.0, max_samples: int = 900, max_requests: int = 200):
        self.interval_s = interval_s
        self.samples: deque = deque(maxlen=max_samples)
        self.requests: deque = deque(maxlen=max_requests)
        self.engine: dict = {}
        self._pid_fn = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- configuration (called by the server once the backend is known)
    def set_engine(
        self,
        name: str,
        kind: str,
        kv: dict | None,
        ctx_alloc_tokens: int = 0,
        max_concurrency: int = 1,
        pid_fn=None,
    ) -> None:
        with self._lock:
            self.engine = {
                "name": name,
                "kind": kind,
                "kv": kv,
                "ctx_alloc_tokens": ctx_alloc_tokens,
                "max_concurrency": max_concurrency,
                "kv_buffer": "preallocated for the full context"
                if kind == "llamacpp"
                else "grows with the tokens in context",
            }
            self._pid_fn = pid_fn

    # ---- recording
    def sample(self) -> dict:
        s: dict = {"t": time.time(), "ram_used_gb": round(psutil.virtual_memory().used / GB, 2)}
        g = gpu_memory()
        if g:
            s["vram_used_gb"], s["vram_total_gb"] = round(g[0], 2), round(g[1], 2)
        pid = self._pid_fn() if self._pid_fn else None
        if pid:
            try:
                s["engine_rss_gb"] = round(psutil.Process(pid).memory_info().rss / GB, 2)
            except psutil.Error:
                pass
        with self._lock:
            self.samples.append(s)
        return s

    def record_request(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        seconds: float,
        model: str = "",
        stream: bool = False,
    ) -> dict:
        bpt = (self.engine.get("kv") or {}).get("bytes_per_token")
        g = gpu_memory()
        r = {
            "t": time.time(),
            "model": model,
            "stream": stream,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "context_tokens": prompt_tokens + completion_tokens,
            "seconds": round(seconds, 3),
            "tok_per_s": round(completion_tokens / seconds, 2) if seconds > 0 and completion_tokens else None,
            "kv_used_mb": round((prompt_tokens + completion_tokens) * bpt / MB, 2) if bpt else None,
            "vram_used_gb": round(g[0], 2) if g else None,
        }
        with self._lock:
            self.requests.append(r)
        return r

    # ---- reading
    def snapshot(self) -> dict:
        with self._lock:
            eng = dict(self.engine)
            samples, requests = list(self.samples), list(self.requests)
        kv = eng.get("kv") or {}
        alloc = eng.get("ctx_alloc_tokens") or max([r["context_tokens"] for r in requests], default=0) or 2048
        curve = kv_curve(kv["bytes_per_token"], alloc) if kv.get("bytes_per_token") else []
        return {
            "engine": eng,
            "kv_curve": curve,
            "kv_alloc_mb": round(alloc * kv["bytes_per_token"] / MB, 1)
            if kv.get("bytes_per_token")
            else None,
            "samples": samples,
            "requests": requests,
        }

    # ---- background sampler
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(self.interval_s):
                self.sample()

        self._thread = threading.Thread(target=loop, name="tinyforge-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def dumps(snapshot: dict) -> str:
    return json.dumps(snapshot)
