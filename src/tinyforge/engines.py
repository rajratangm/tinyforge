"""Serving engines behind the OpenAI-compatible API: a generic upstream proxy and a managed llama.cpp server.

`ProxyBackend` forwards chat requests to any OpenAI-compatible upstream and yields the content deltas, so our
router keeps ownership of auth, guardrails, metrics and limits. `LlamaCppBackend` additionally starts and
stops `llama-server` for you:

* bound to 127.0.0.1 only, with a random per-launch API key so other local processes cannot use it;
* started lazily on the first request (importing this module never spawns anything) and stopped at exit;
* GPU layers chosen from free VRAM (`choose_ngl`) unless you pass a number;
* `parallel` slots give continuous-batching style concurrency, advertised to the router as `max_concurrency`.

Select with `TINYFORGE_ENGINE=llamacpp` plus `TINYFORGE_GGUF` (and optionally `TINYFORGE_LORA_GGUF`).

Limits: the engine does not coordinate with training jobs for the GPU (the router only refuses requests
while a job is running); a crashed server is restarted on the next request, not mid-request; `/v1/models`
reports the GGUF file name, not the original Hugging Face id.
"""
from __future__ import annotations

import atexit
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

GB = 1024**3
ROOT = Path(__file__).resolve().parents[2]


class UpstreamError(RuntimeError):
    """The serving engine failed or refused the request."""


class ProxyBackend:
    """Backend protocol implementation that talks to an OpenAI-compatible server over HTTP."""

    max_concurrency = 1

    def __init__(self, base_url: str, model: str = "", api_key: str = "", name: str = "upstream",
                 timeout: float = 600.0):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key, self.name, self.timeout = api_key, name, timeout

    def _url(self) -> str:
        return self.base_url

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def stream(self, messages: list[dict], max_tokens: int, temperature: float,
               top_p: float) -> Iterator[str]:
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                "stream": True}
        if self.model:
            body["model"] = self.model
        try:
            url = f"{self._url()}/v1/chat/completions"
            with httpx.stream("POST", url, json=body, headers=self._headers(), timeout=self.timeout) as r:
                if r.status_code != 200:
                    detail = r.read().decode("utf-8", "replace")[:200]
                    raise UpstreamError(f"engine returned HTTP {r.status_code}: {detail}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except httpx.HTTPError as e:
            raise UpstreamError(f"engine unreachable: {e}") from e

    def count_tokens(self, text: str) -> int:
        try:
            r = httpx.post(f"{self._url()}/tokenize", json={"content": text}, headers=self._headers(),
                           timeout=10)
            r.raise_for_status()
            return len(r.json()["tokens"])
        except Exception:  # noqa: BLE001 - usage numbers must never fail a request
            return len(text.split())


def free_vram_bytes() -> int:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        out = subprocess.run([exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0]) * 1024 * 1024
    except Exception:  # noqa: BLE001
        return 0


def choose_ngl(model_bytes: int, free_vram: int, n_layers: int = 32, reserve: float = 0.45 * GB) -> int:
    """GPU layers for a GGUF: all if it fits with KV/compute headroom, else a proportional share, else 0.

    Measured: Llama-3.1-8B Q4_K_M (4.9 GB, 32 layers) ran well with 24 layers on a 4 GB card; this rule picks
    19 there (a deliberately safe choice that leaves room for the desktop and a longer context).
    """
    if free_vram <= 0:
        return 0
    if model_bytes + reserve <= free_vram:
        return 999
    usable = free_vram - reserve
    return max(0, int(n_layers * usable / model_bytes))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaCppBackend(ProxyBackend):
    def __init__(self, model: Path, lora: Path | None = None, server_bin: list[str] | None = None,
                 ngl: int | str = "auto", ctx: int = 2048, parallel: int = 2, startup_timeout: float = 240.0):
        self.model_path, self.lora_path = Path(model), Path(lora) if lora else None
        self.server_bin, self.ngl, self.ctx, self.parallel = server_bin, ngl, ctx, max(1, parallel)
        self.startup_timeout = startup_timeout
        self.max_concurrency = self.parallel
        self.proc: subprocess.Popen | None = None
        self.log_path = Path(tempfile.gettempdir()) / f"tinyforge-llama-{os.getpid()}.log"
        self._lock = threading.Lock()
        super().__init__("", api_key=secrets.token_urlsafe(24),
                         name=self.model_path.stem + ("+lora" if lora else ""))
        atexit.register(self.stop)

    def _url(self) -> str:
        self._ensure()
        return self.base_url

    @staticmethod
    def locate_server() -> list[str]:
        explicit = os.environ.get("TINYFORGE_LLAMA_SERVER", "").strip()
        exe = ".exe" if sys.platform == "win32" else ""
        bundled = ROOT / "tools" / "llama.cpp" / "bin" / f"llama-server{exe}"
        cand = explicit or shutil.which("llama-server") or str(bundled)
        if not Path(cand).exists() and not shutil.which(cand):
            raise FileNotFoundError("llama-server not found: set TINYFORGE_LLAMA_SERVER to its path")
        return [cand]

    def build_cmd(self, port: int) -> list[str]:
        ngl = self.ngl
        if ngl == "auto":
            ngl = choose_ngl(self.model_path.stat().st_size, free_vram_bytes())
        cmd = (self.server_bin or self.locate_server()) + [
            "-m", str(self.model_path), "--host", "127.0.0.1", "--port", str(port),
            "-c", str(self.ctx * self.parallel), "-np", str(self.parallel), "-ngl", str(ngl),
            "--api-key", self.api_key, "--no-webui"]
        if self.lora_path:
            cmd += ["--lora", str(self.lora_path)]
        return cmd

    def _ensure(self) -> None:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            if not self.model_path.exists():
                raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
            if self.lora_path and not self.lora_path.exists():
                raise FileNotFoundError(f"LoRA GGUF not found: {self.lora_path}")
            port = _free_port()
            cmd = self.build_cmd(port)
            logf = self.log_path.open("wb")
            self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            self.base_url = f"http://127.0.0.1:{port}"
            deadline = time.time() + self.startup_timeout
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise UpstreamError("llama-server exited during startup: " + self._log_tail())
                try:
                    if httpx.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            self.stop()
            raise UpstreamError("llama-server did not become healthy in time: " + self._log_tail())

    def _log_tail(self) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            return " | ".join(text.strip().splitlines()[-6:])
        except OSError:
            return "(no log)"

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def make_backend(env: dict | None = None, default=None):
    """Pick the serving engine from the environment (default: the supplied in-process backend)."""
    e = os.environ if env is None else env
    engine = e.get("TINYFORGE_ENGINE", "hf").strip().lower()
    if engine == "hf":
        return default
    if engine == "llamacpp":
        gguf = e.get("TINYFORGE_GGUF", "").strip()
        if not gguf:
            raise ValueError("TINYFORGE_ENGINE=llamacpp needs TINYFORGE_GGUF (path to the base GGUF)")
        lora = e.get("TINYFORGE_LORA_GGUF", "").strip() or None
        ngl = e.get("TINYFORGE_NGL", "auto").strip()
        return LlamaCppBackend(Path(gguf), Path(lora) if lora else None,
                               ngl=ngl if ngl == "auto" else int(ngl),
                               ctx=int(e.get("TINYFORGE_CTX", "2048")),
                               parallel=int(e.get("TINYFORGE_PARALLEL", "2")))
    raise ValueError(f"TINYFORGE_ENGINE must be hf or llamacpp, got {engine!r}")
