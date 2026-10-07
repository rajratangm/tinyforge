"""FastAPI backend. Jobs are CLI subprocesses (crash isolation, GPU freed on exit), queued in SQLite.

Security: every /api/* route needs `Authorization: Bearer $TINYFORGE_API_TOKEN`. With no token configured the
API refuses to serve unless TINYFORGE_AUTH=off is set explicitly (local development only). /healthz and the
static UI page stay open; the UI asks for the token and sends it with each call.

Network hardening (see netsec.py): security headers + CSP on every response, a 1 MiB request-body cap, CORS
off unless TINYFORGE_CORS_ORIGINS lists explicit origins, 429 after repeated wrong tokens from one client, and
the interactive /docs, /redoc and /openapi.json pages off unless TINYFORGE_DOCS=on.
"""

from __future__ import annotations

import hmac
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__, hardware
from .jobstore import JobStore
from .metrics import CONTENT_TYPE as METRICS_CONTENT_TYPE
from .metrics import LOG_SCAN_LINES
from .metrics import render as render_metrics
from .netsec import AuthFailureLimiter, NetSettings
from .netsec import install as install_net_security

ROOT = Path.cwd()
UI_DIR = Path(__file__).parent / "ui"

_models: dict[str, tuple] = {}  # ckpt path -> (model, tokenizer)
_ft_models: dict[str, tuple] = {}  # "m" -> (peft model, tokenizer, device)
_lock = threading.Lock()


# ---------------------------------------------------------------- auth


_net = NetSettings.from_env()  # a bad TINYFORGE_CORS_ORIGINS etc. fails at startup with a clear message
_auth_limiter = AuthFailureLimiter(_net.auth_fail_limit, _net.auth_fail_window_s)


def require_auth(request: Request) -> None:
    if os.environ.get("TINYFORGE_AUTH", "").lower() == "off":
        return
    token = os.environ.get("TINYFORGE_API_TOKEN", "")
    if not token:
        raise HTTPException(
            503,
            "API token not configured: set TINYFORGE_API_TOKEN "
            "(or TINYFORGE_AUTH=off for local development only).",
        )
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    supplied = supplied.strip()
    # Only requests that present a bearer token count as guesses; a request with no token (how the UI finds
    # out it needs one) is neither limited nor counted. The client key is the peer address uvicorn reports.
    guess = scheme.lower() == "bearer" and bool(supplied)
    client = request.client.host if request.client else "unknown"
    if guess and (retry := _auth_limiter.blocked(client)) is not None:
        raise HTTPException(
            429,
            "Too many failed authentication attempts; try again later.",
            headers={"Retry-After": str(retry)},
        )
    # compare as bytes: compare_digest raises on non-ASCII str, and runs in constant time
    if not guess or not hmac.compare_digest(supplied.encode(), token.encode()):
        if guess:
            _auth_limiter.record_failure(client)
        raise HTTPException(401, "Missing or invalid API token.", headers={"WWW-Authenticate": "Bearer"})


api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])


# ---------------------------------------------------------------- job queue


def _release_gpu() -> None:
    """Drop cached chat models so a training job gets the whole GPU."""
    _models.clear()
    _ft_models.clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


StageRunner = Callable[[str, list[str], Callable[[str], None]], int]  # (job_id, argv, emit_line) -> exit code
_MEM_LOGS = 20  # finished jobs whose logs stay in memory; older ones are served from their log file


class JobManager:
    """Runs queued jobs one at a time on a worker thread. Durable state lives in the JobStore."""

    def __init__(
        self,
        store: JobStore,
        stage_runner: StageRunner | None = None,
        before_job: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self._run_stage = stage_runner or self._run_subprocess
        self._before_job = before_job or _release_gpu
        self.logs: dict[str, list[str]] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: str | None = None
        self._proc: subprocess.Popen | None = None
        self._cancel_requested: str | None = None

    def start(self) -> None:
        if self._thread:
            return
        self.store.recover()  # running -> interrupted; queued jobs are kept and will run
        self._thread = threading.Thread(target=self._loop, daemon=True, name="job-runner")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()  # job stays 'running' in the DB; the next start marks it 'interrupted'
        if self._thread:
            self._thread.join(timeout=10)

    def submit(self, kind: str, stages: list[list[str]]) -> str:
        job_id = self.store.enqueue(kind, stages)
        self._wake.set()
        return job_id

    def cancel(self, job_id: str) -> bool:
        if self.store.cancel_queued(job_id):
            return True
        if self._current == job_id:
            self._cancel_requested = job_id
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            return True
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.claim_next()
            if job is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            self._run(job)

    def _run(self, job: dict) -> None:
        job_id = job["id"]
        log = self.logs.setdefault(job_id, [])
        self._current, self._proc, self._cancel_requested = job_id, None, None
        status, code = "done", 0
        try:
            with open(self.store.log_path(job_id), "a", encoding="utf-8") as fh:

                def emit(line: str) -> None:
                    log.append(line)
                    fh.write(line + "\n")
                    fh.flush()

                try:
                    self._before_job()
                    for argv in job["stages"]:
                        code = self._run_stage(job_id, argv, emit)
                        if self._cancel_requested == job_id:
                            status = "cancelled"
                            break
                        if code != 0:
                            status = "failed"
                            break
                except Exception as e:  # a runner bug must not kill the worker thread
                    status, code = "failed", -1
                    emit(f"runner error: {type(e).__name__}: {e}")
        finally:
            self.store.finish(job_id, status, code, log)
            self._current, self._proc = None, None
            finished = [j["id"] for j in self.store.list(_MEM_LOGS * 2) if j["status"] != "running"]
            for old in finished[:-_MEM_LOGS]:
                self.logs.pop(old, None)

    def _run_subprocess(self, job_id: str, argv: list[str], emit: Callable[[str], None]) -> int:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tinyforge", *argv],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._proc = proc
        for line in proc.stdout:  # type: ignore[union-attr]
            emit(line.rstrip())
        return proc.wait()

    def job_log(self, job_id: str) -> list[str] | None:
        """In-memory list for live/recent jobs (the SSE stream tails it), else None."""
        return self.logs.get(job_id)


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def _db_path() -> Path:
    return Path(os.environ.get("TINYFORGE_DB") or ROOT / "runs" / "jobs.db")


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager(JobStore(_db_path()))
            _manager.start()
        return _manager


def configure(
    db_path: Path,
    stage_runner: StageRunner | None = None,
    before_job: Callable[[], None] | None = None,
    autostart: bool = True,
) -> JobManager:
    """Replace the manager (used by tests; also a hook for embedding). The previous one is stopped."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.stop()
        _manager = JobManager(JobStore(db_path), stage_runner, before_job)
        if autostart:
            _manager.start()
        return _manager


def _busy() -> bool:
    return _manager is not None and _manager.store.has_running()


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_manager()  # recover interrupted jobs and resume the queue at server start, not at first request
    telemetry.start()
    yield
    telemetry.stop()
    if _manager is not None:
        _manager.stop()


app = FastAPI(
    title="tinyforge",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if _net.docs_enabled else None,
    redoc_url="/redoc" if _net.docs_enabled else None,
    openapi_url="/openapi.json" if _net.docs_enabled else None,
)
_ui_page = UI_DIR / "index.html"
install_net_security(app, _net, _ui_page.read_bytes() if _ui_page.is_file() else None)


class TrainRequest(BaseModel):
    preset: str = Field("micro", pattern="^(nano|micro|small|base)$")
    steps: int = Field(2000, ge=1, le=1_000_000)
    block_size: int = Field(256, ge=16, le=4096)


class GenRequest(BaseModel):
    prompt: str = Field("ROMEO:", max_length=4000)
    run: str = Field("micro", pattern=r"^[A-Za-z0-9_.-]+$")  # no path traversal
    max_new: int = Field(200, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0, le=2)
    top_k: int = Field(50, ge=0, le=1000)
    int8: bool = False


@api.get("/telemetry")
def get_telemetry():
    """Memory over time, KV-cache math and per-request records for the dashboard."""
    _register_engine()
    return telemetry.snapshot()


@api.get("/hardware")
def get_hardware():
    from . import deps

    hw = hardware.probe()
    return {
        "hardware": hw.to_dict(),
        "diagnostics": hardware.diagnose(hw).to_list(),
        "components": deps.check_components(),
    }


def _backends_installed() -> tuple[str, ...]:
    from . import deps

    have = {c["name"] for c in deps.check_components() if c["installed"]}
    return tuple(b for b, need in (("native", "finetune"), ("soup", "soup")) if need in have)


@api.get("/methods/suggest")
def api_methods_suggest(
    params_b: float = Query(8.0, gt=0.01, le=1000),
    data: str = Query("sft", pattern="^(sft|preference)$"),
    prefer: str = Query("quality", pattern="^(quality|fit)$"),
):
    """Which fine-tuning methods fit this machine for this model size (`tinyforge methods suggest`)."""
    from . import memtiers, methods

    h = memtiers.probe_hierarchy(measure=False)
    have = _backends_installed()
    recs, ctx = methods.recommend_methods(h, params_b, data, have or ("native", "soup"), prefer)
    ctx["installed_backends"] = list(have)
    return {"context": ctx, "methods": [r.to_dict() for r in recs]}


@api.get("/bench/suggest")
def api_bench_suggest(
    params_b: float = Query(8.0, gt=0.01, le=1000),
    quant: str = Query("4bit", pattern="^(4bit|fp16)$"),
    goal: str = Query("general", pattern="^(general|forgetting|reasoning|instruction|sql|safety)$"),
    minutes: float = Query(30.0, ge=1, le=1000),
    measured_tps: float = Query(0.0, ge=0),
):
    """Benchmarks and sample sizes that fit this machine and time budget (see `tinyforge bench suggest`)."""
    from . import bench, memtiers

    h = memtiers.probe_hierarchy(measure=False)
    recs, ctx = bench.recommend(h, params_b, quant, minutes, goal, measured_tps or None)
    return {"context": ctx, "recommendations": [r.to_dict() for r in recs]}


@api.post("/jobs/pipeline")
def start_pipeline(req: TrainRequest):
    data = ["data", "prepare", "--out", "data/tinyshakespeare"]
    train = [
        "train",
        "--preset",
        req.preset,
        "--steps",
        str(req.steps),
        "--block-size",
        str(req.block_size),
        "--run-dir",
        f"runs/{req.preset}",
        "--json",
    ]
    ev = ["eval", "--ckpt", f"runs/{req.preset}/best.pt", "--json"]
    return {"id": get_manager().submit("pipeline", [data, train, ev])}


class FTRequest(BaseModel):
    base_model: str = Field("HuggingFaceTB/SmolLM2-360M-Instruct", pattern=r"^[\w.-]+/[\w.-]+$")
    steps: int = Field(150, ge=1, le=100_000)
    limit: int = Field(3000, ge=100, le=200_000)


class FTGenRequest(BaseModel):
    prompt: str = Field(max_length=4000)
    max_new: int = Field(200, ge=1, le=1000)
    base: bool = False  # True: bypass the adapter to compare with the original model


@api.post("/jobs/finetune")
def start_finetune(req: FTRequest):
    data = ["ft", "data", "--base-model", req.base_model, "--limit", str(req.limit)]
    train = ["ft", "train", "--base-model", req.base_model, "--steps", str(req.steps), "--json"]
    return {"id": get_manager().submit("finetune", [data, train, ["ft", "eval", "--json"]])}


@api.post("/ft/generate")
def ft_generate(req: FTGenRequest):
    if _busy():
        raise HTTPException(409, "GPU is busy with a running job.")
    run_dir = ROOT / "runs" / "ft"
    if not (run_dir / "best").exists():
        raise HTTPException(404, "No fine-tuned model yet.")
    with _lock:
        if "m" not in _ft_models:
            from peft import PeftModel

            from .finetune import FTConfig, load_base, load_tokenizer

            cfg = FTConfig.model_validate_json((run_dir / "ft_config.json").read_text())
            hw = hardware.probe()
            base = load_base(cfg.base_model, cfg.quant, hw.bf16_supported, hw.device)
            _ft_models["m"] = (
                PeftModel.from_pretrained(base, run_dir / "best"),
                load_tokenizer(cfg.base_model),
                hw.device,
            )
        model, tok, device = _ft_models["m"]
        from .ft_eval import _gen

        if req.base:
            with model.disable_adapter():
                return {"text": _gen(model, tok, req.prompt, device, req.max_new)}
        return {"text": _gen(model, tok, req.prompt, device, req.max_new)}


@api.get("/jobs")
def list_jobs():
    keys = ("id", "kind", "status", "code", "created", "started", "finished")
    return [{k: j[k] for k in keys} for j in get_manager().store.list()]


@api.get("/jobs/{job_id}/events")
def job_events(job_id: str):
    mgr = get_manager()
    job = mgr.store.get(job_id) if len(job_id) == 8 else None
    if not job:
        raise HTTPException(404)

    def stream():
        i = 0
        while True:
            # Read the status BEFORE draining: a terminal status means every line is already written.
            status = mgr.store.get(job_id)["status"]  # type: ignore[index]
            active = status in ("queued", "running")
            lines = mgr.job_log(job_id)  # live list; appears once a queued job starts
            if lines is None:
                lines = [] if active else mgr.store.read_log(job_id)
            while i < len(lines):
                yield f"data: {json.dumps(lines[i])}\n\n"
                i += 1
            if not active:
                yield f"event: end\ndata: {status}\n\n"
                return
            time.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@api.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    if len(job_id) != 8 or not get_manager().cancel(job_id):
        raise HTTPException(404)
    return {"ok": True}


@api.get("/runs")
def list_runs():
    runs = []
    for d in sorted((ROOT / "runs").glob("*")) if (ROOT / "runs").exists() else []:
        if not d.is_dir() or d.name.startswith("_"):  # skips runs/_jobs (queue logs)
            continue
        ev = d / "eval.json"
        is_ft = (d / "ft_config.json").exists()
        runs.append(
            {
                "name": d.name,
                "kind": "finetune" if is_ft else "scratch",
                "has_model": (d / "best").exists() if is_ft else (d / "best.pt").exists(),
                "eval": json.loads(ev.read_text()) if ev.exists() else None,
            }
        )
    return runs


@api.post("/generate")
def api_generate(req: GenRequest):
    from . import infer
    from .data import load_tokenizer
    from .train import load_model

    if _busy():
        raise HTTPException(409, "GPU is busy with a running job.")
    ckpt = ROOT / "runs" / req.run / "best.pt"
    if not ckpt.exists():
        raise HTTPException(404, f"No trained model for run '{req.run}'.")
    key = f"{ckpt}:{req.int8}"
    if key not in _models:
        model, _ = load_model(ckpt, hardware.probe().device)
        _models[key] = (
            infer.quantize_int8(model) if req.int8 else model,
            load_tokenizer(ROOT / "data" / "tinyshakespeare"),
        )
    model, tok = _models[key]
    with _lock:
        text = infer.generate_text(
            model, tok, req.prompt, max_new=req.max_new, temperature=req.temperature, top_k=req.top_k
        )
    return {"text": text}


app.include_router(api)

from .engines import make_backend  # noqa: E402
from .guardrails import COUNTS as GUARD_COUNTS  # noqa: E402
from .guardrails import GuardConfig, build_filters  # noqa: E402
from .openai_api import HFBackend, build_router  # noqa: E402
from .telemetry import Telemetry  # noqa: E402

# TINYFORGE_ENGINE=llamacpp swaps the in-process model for a managed llama-server; tests replace this
_backend = make_backend(os.environ, HFBackend(ROOT / "runs" / "ft"))
_guard_in, _guard_out = build_filters(GuardConfig.from_env())  # bad env values fail at startup
telemetry = Telemetry()


def _on_request(r: dict) -> None:
    _register_engine()  # so KV figures exist even if nobody has opened the dashboard yet
    telemetry.record_request(**r)


def _register_engine() -> None:
    """Describe the serving backend to the telemetry (KV-cache math needs its layer/head numbers)."""
    info = getattr(_backend, "telemetry_info", None)
    if info is None:
        return
    cur = telemetry.engine
    if cur.get("kv") is not None and cur.get("name") == getattr(_backend, "name", None):
        return  # already described (the GGUF header read is not free); retry until the KV numbers are known
    telemetry.set_engine(**info())


app.include_router(
    build_router(
        require_auth,
        lambda: _backend,
        _busy,
        _guard_in,
        _guard_out,
        on_request=_on_request,
        on_reject=telemetry.record_reject,
    )
)


_METRICS_JOB_LIMIT = 100_000  # JobStore.list is capped; the table has no retention yet, so counts stop here


@app.get("/metrics", dependencies=[Depends(require_auth)])
def metrics():
    """Prometheus scrape endpoint. Same bearer token as /api/* (use `authorization.credentials_file`)."""
    mgr = get_manager()
    jobs = mgr.store.list(_METRICS_JOB_LIMIT)
    running = next((j for j in jobs if j["status"] == "running"), None)
    lines = (mgr.job_log(running["id"]) or []) if running else []
    import psutil

    vm = psutil.virtual_memory()
    host = {"ram_total": vm.total, "ram_available": vm.available, "swap_used": psutil.swap_memory().used}
    return Response(
        render_metrics(__version__, jobs, list(lines[-LOG_SCAN_LINES:]), host, dict(GUARD_COUNTS)),
        media_type=METRICS_CONTENT_TYPE,
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": __version__}


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")
