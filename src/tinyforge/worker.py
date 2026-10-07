"""`tinyforge worker run`: the Python half of the worker contract (spec/worker-contract.md).

Reads a TrainingJob (tinyforge.dev/v1alpha1), validates it, maps it onto FTConfig and runs finetune.train.
stdout carries only JSON-lines events; human text goes to stderr. Exit codes follow the contract:
0 ok | 2 spec invalid/unsupported | 3 gate failed | 4 does not fit / out of memory | 5 diverged |
75 preempted.

Known gaps (worker v1): spec.model.revision and data.format are accepted but not passed through; exports other
than "adapter" are not produced; `artifact` and `heartbeat` events are not emitted; gate metrics other than
val_loss are computed by running the existing eval commands after training, which is untested end to end.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

worker_app = typer.Typer(help="Run a TrainingJob as a worker (used by the node agent).", no_args_is_help=True)

EXIT_OK, EXIT_CRASH, EXIT_SPEC, EXIT_GATE, EXIT_FIT, EXIT_DIVERGED, EXIT_PREEMPTED = 0, 1, 2, 3, 4, 5, 75
SCHEMA_ENV = "TINYFORGE_JOBSPEC_SCHEMA"
_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


class SpecError(Exception):
    """The spec is invalid or asks for something worker v1 does not support (exit 2)."""


class Preempted(BaseException):
    """Raised from the SIGTERM handler. BaseException so no `except Exception` in training swallows it."""


def _schema_path() -> Path:
    if os.environ.get(SCHEMA_ENV):
        return Path(os.environ[SCHEMA_ENV])
    # Source checkout layout. A wheel/Docker install must set TINYFORGE_JOBSPEC_SCHEMA (not packaged yet).
    return Path(__file__).resolve().parents[2] / "spec" / "jobspec.v1alpha1.schema.json"


def load_spec(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate specs: pip install jsonschema") from exc
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SpecError(f"cannot read spec {path}: {exc}") from exc
    sp = _schema_path()
    if not sp.exists():
        raise RuntimeError(f"JobSpec schema not found at {sp}; set {SCHEMA_ENV}")
    schema = json.loads(sp.read_text(encoding="utf-8"))
    errs = sorted(jsonschema.Draft202012Validator(schema).iter_errors(doc),
                  key=lambda e: [str(p) for p in e.absolute_path])
    if errs:
        raise SpecError("; ".join(f"/{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in errs[:5]))
    return doc


def check_supported(doc: dict) -> None:
    s = doc["spec"]
    res = s.get("resources", {})
    if s["method"] == "full":
        raise SpecError("method=full is not implemented by worker v1")
    if res.get("nodes", 1) > 1:
        raise SpecError("nodes>1 is not supported by worker v1 (single node only)")
    if res.get("gpus", 1) > 1:
        raise SpecError("gpus>1 is not supported by worker v1 (0 or 1 GPU)")
    if s.get("backend", "native") == "soup" and res.get("gpus", 1) < 1:
        raise SpecError("backend=soup needs a GPU (resources.gpus >= 1)")


def to_ft_config(doc: dict, out: Path):
    """Map spec -> FTConfig. Only fields in the spec are set; FTConfig defaults equal schema defaults."""
    from .finetune import FTConfig

    s = doc["spec"]
    h, d, ck = s.get("hyperparameters", {}), s["data"], s.get("checkpoint", {})
    kw: dict[str, Any] = {
        "base_model": s["model"]["base"], "run_dir": out, "data_dir": out / "data",
        "quant": "4bit" if s["method"] == "qlora" else "none",
    }
    pairs = [("max_steps", h, "maxSteps"), ("lr", h, "learningRate"), ("warmup_steps", h, "warmupSteps"),
             ("grad_clip", h, "gradClip"), ("batch_size", h, "batchSize"), ("grad_accum", h, "gradAccum"),
             ("lora_r", h, "loraR"), ("lora_alpha", h, "loraAlpha"), ("lora_dropout", h, "loraDropout"),
             ("grad_checkpointing", h, "gradCheckpointing"), ("seed", h, "seed"),
             ("max_len", d, "maxLen"), ("eval_interval", ck, "everySteps")]
    for field, src, key in pairs:
        if key in src:
            kw[field] = src[key]
    return FTConfig(**kw)


def _data_ready(data_dir: Path) -> bool:
    return all((data_dir / f).exists() for f in ("train.jsonl", "val.jsonl", "meta.json"))


def evaluate_gates(gates: list[dict], summary: dict, out: Path, data_format: str, emit) -> bool:
    """Evaluate spec.gates; emit a `gate` event each. A gate that cannot be evaluated FAILS (fail closed)."""
    cache: dict[str, dict] = {}

    def observed(metric: str) -> float:
        if metric == "val_loss":
            return float(summary["best_val_loss"])
        if metric in ("gain_pct", "forgetting_pct"):
            if "ft_eval" not in cache:
                from . import ft_eval

                cache["ft_eval"] = ft_eval.evaluate(out, out / "data", True)
            return float(cache["ft_eval"]["improvement_pct" if metric == "gain_pct" else "forgetting_pct"])
        if metric == "task_exact_match_gain_pts":
            if data_format != "sql-create-context":
                raise ValueError(f"no task evaluator for data.format={data_format!r}")
            if "ft_task" not in cache:
                from . import ft_task

                cache["ft_task"] = ft_task.evaluate_task(out, "sql", 100, 0.0)
            return float(cache["ft_task"]["gain_pts_lenient_em"])
        raise ValueError(f"unknown metric {metric!r}")

    ok = True
    for g in gates:
        try:
            val = observed(g["metric"])
            passed = bool(_OPS[g["op"]](val, g["value"]))
            emit("gate", metric=g["metric"], op=g["op"], value=g["value"], observed=val, passed=passed)
        except Exception as exc:  # noqa: BLE001 - any evaluation failure must fail the gate, not crash the job
            passed = False
            emit("gate", metric=g["metric"], op=g["op"], value=g["value"], observed=None, passed=False,
                 reason=f"could not evaluate: {exc}")
        ok &= passed
    return ok


MODEL_CACHE = Path(os.environ.get("TINYFORGE_MODEL_DIR", "")
                   or Path.home() / ".cache" / "tinyforge" / "models")


def _run_soup(doc: dict, cfg, out: Path, emit, say) -> dict | int:
    """Run the Soup backend. Returns the finished-event summary, or an exit code on failure."""
    from . import memtiers, soup_backend

    for w in memtiers.ram_pressure_warnings(memtiers.probe_hierarchy(measure=False)):
        emit("diagnostic", code="WK010", level="warn", message=w, fix="")
    revision = doc["spec"]["model"].get("revision")
    model_dir = soup_backend.ensure_local_model(cfg.base_model, MODEL_CACHE, revision)
    soup_backend.write_run_config(out, model_dir, doc)
    summary: dict = {}

    def capture(kind: str, /, **kw) -> None:
        if kind == "finished":
            summary.update(kw)
        emit(kind, **kw)

    code, reason = soup_backend.run_soup(doc, out, capture, model_dir)
    if code == 4:
        return EXIT_FIT
    if code:
        say(f"soup backend failed: {reason}")
        return EXIT_CRASH
    return summary


def run_job(spec_path: Path, out: Path, dry_run: bool = False) -> int:
    t_emit = sys.stdout

    def emit(kind: str, /, **kw) -> None:
        print(json.dumps({"event": kind, "t": time.time(), **kw}), file=t_emit, flush=True)

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    try:
        doc = load_spec(spec_path)
        check_supported(doc)
        cfg = to_ft_config(doc, out)
    except SpecError as exc:
        say(f"spec invalid: {exc}")
        emit("failed", reason=f"spec invalid: {exc}")
        return EXIT_SPEC
    except RuntimeError as exc:  # missing dependency / schema file: an environment fault, not a bad spec
        say(str(exc))
        emit("failed", reason=str(exc))
        return EXIT_CRASH

    if dry_run:
        print(json.dumps(cfg.model_dump(mode="json")), file=t_emit, flush=True)
        return EXIT_OK

    s = doc["spec"]
    notes = []
    if s["model"].get("revision"):
        notes.append(("WK001", "model.revision is not passed to from_pretrained yet; latest is used."))
    skipped = [e for e in s.get("export", ["adapter"]) if e != "adapter"]
    if skipped:
        notes.append(("WK002", f"exports {skipped} are not produced by worker v1; only the adapter."))
    for code, msg in notes:
        emit("diagnostic", code=code, level="warn", message=msg, fix="")

    def on_sigterm(signum, frame):  # noqa: ARG001
        raise Preempted

    signal.signal(signal.SIGTERM, on_sigterm)
    out.mkdir(parents=True, exist_ok=True)
    last_failed: dict[str, str] = {}

    def on_event(e: dict) -> None:
        if e.get("event") == "failed":
            last_failed["reason"] = str(e.get("reason", ""))
        print(json.dumps(e), file=t_emit, flush=True)

    try:
        from . import finetune

        if not _data_ready(cfg.data_dir):
            from . import ft_data

            d = s["data"]
            meta, rep = ft_data.prepare(d["source"], cfg.data_dir, cfg.base_model, d.get("limit", 3000),
                                        d.get("valPercent", 5), cfg.max_len)
            for diag in rep.to_list():
                emit("diagnostic", **diag)
            if rep.has_errors:
                emit("failed", reason="data preparation failed")
                return EXIT_CRASH
        if s.get("backend", "native") == "soup":
            summary = _run_soup(doc, cfg, out, emit, say)
            if isinstance(summary, int):
                return summary
        else:
            summary = finetune.train(cfg, on_event)
    except Preempted:
        # Weaker than the contract: we stop immediately instead of finishing the step. Checkpoints are written
        # atomically, so the retry resumes from the last one (at most checkpoint.everySteps of work is lost).
        emit("failed", reason="preempted")
        return EXIT_PREEMPTED
    except RuntimeError as exc:
        reason = last_failed.get("reason", "")
        say(str(exc))
        if "out of memory" in reason or "did not fit" in reason:
            return EXIT_FIT
        if "diverged" in reason:
            return EXIT_DIVERGED
        emit("failed", reason=str(exc))
        return EXIT_CRASH

    if not evaluate_gates(s.get("gates", []), summary, out, s["data"].get("format", "text"), emit):
        emit("failed", reason="gate failed")
        return EXIT_GATE
    return EXIT_OK


@worker_app.command("run")
def run(
    spec: Annotated[Path, typer.Option(help="TrainingJob YAML/JSON file.")],
    out: Annotated[Path, typer.Option(help="Job scratch/output directory.")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate, print FTConfig, exit.")] = False,
) -> None:
    """Run (or --dry-run) a TrainingJob. Exit codes: see spec/worker-contract.md."""
    raise typer.Exit(run_job(spec, out, dry_run))
