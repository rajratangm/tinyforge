"""Benchmark catalog and a hardware-aware recommender.

Which benchmarks make sense depends on the machine: a loglikelihood benchmark (MMLU, ARC, HellaSwag) needs
the model resident on the GPU, while generative ones (GSM8K, IFEval, our SQL execution check) can run against
a llama.cpp/OpenAI-compatible server even when the model is larger than VRAM, but cost decode time.
`recommend` turns the probed memory hierarchy + model size + a time budget into a ranked, explained plan with
a sample size per benchmark; `lm_eval_cmd` builds the matching `lm-evaluation-harness` command.

Honest limits: every cost figure is an ESTIMATE from a simple model (decode speed from memory bandwidth,
prefill assumed a fixed multiple of decode), so pass a measured tok/s (`measured_tps`) when you have one.
Sample sizes are sub-samples: a 100-item MMLU slice has roughly +-5 points of noise, so compare base vs
tuned on the SAME items and treat small differences as noise. Benchmarks measure what they measure: a tuned
SQL model should not be expected to move MMLU; use them to catch forgetting, not to show gains.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import asdict, dataclass, field

from .memtiers import GB, NF4_BYTES_PER_PARAM, Hierarchy

PREFILL_MULT = 8.0  # assumed prefill tok/s = this x decode tok/s (flagged as an assumption)
OVERHEAD_S_PER_ITEM = 0.05  # request/serialisation overhead
ASSUMED_VRAM_BW_GBPS = 150.0  # used only when the GPU bandwidth was not measured (measured here: 151 GB/s)


@dataclass(frozen=True)
class Benchmark:
    name: str
    title: str
    measures: str
    kind: str  # loglik | generate
    task: str  # lm-eval task name, or "builtin:sql-exec"
    items: int  # full test size (approximate)
    prefill_tokens: int  # tokens processed per item (all choices for loglik)
    decode_tokens: int  # generated tokens per item (0 for loglik)
    goals: tuple[str, ...]
    note: str = ""


CATALOG: dict[str, Benchmark] = {
    b.name: b
    for b in [
        Benchmark(
            "mmlu",
            "MMLU (zero-shot)",
            "broad knowledge across 57 subjects",
            "loglik",
            "mmlu",
            14042,
            600,
            0,
            ("general", "forgetting"),
            "the standard 'did fine-tuning damage general knowledge' check",
        ),
        Benchmark(
            "arc_easy",
            "ARC-Easy",
            "grade-school science questions",
            "loglik",
            "arc_easy",
            2376,
            240,
            0,
            ("general", "forgetting"),
        ),
        Benchmark(
            "arc_challenge",
            "ARC-Challenge",
            "harder science questions",
            "loglik",
            "arc_challenge",
            1172,
            240,
            0,
            ("general", "forgetting"),
        ),
        Benchmark(
            "hellaswag",
            "HellaSwag",
            "commonsense sentence completion",
            "loglik",
            "hellaswag",
            10042,
            480,
            0,
            ("general", "forgetting"),
        ),
        Benchmark(
            "winogrande",
            "WinoGrande",
            "commonsense pronoun resolution",
            "loglik",
            "winogrande",
            1267,
            80,
            0,
            ("general", "forgetting"),
        ),
        Benchmark(
            "truthfulqa",
            "TruthfulQA (MC2)",
            "tendency to repeat common falsehoods",
            "loglik",
            "truthfulqa_mc2",
            817,
            480,
            0,
            ("general", "safety"),
        ),
        Benchmark(
            "gsm8k",
            "GSM8K",
            "grade-school math word problems (generation)",
            "generate",
            "gsm8k",
            1319,
            220,
            180,
            ("general", "reasoning", "forgetting"),
        ),
        Benchmark(
            "ifeval",
            "IFEval",
            "following verifiable instructions",
            "generate",
            "ifeval",
            541,
            90,
            220,
            ("instruction", "general"),
            "the right check for chat/instruction tuning",
        ),
        Benchmark(
            "sql-exec",
            "SQL execution accuracy (built in)",
            "text-to-SQL scored by running the query",
            "generate",
            "builtin:sql-exec",
            100,
            150,
            30,
            ("sql",),
            "needs --csv and --test-file (see `data tabular --hard`)",
        ),
    ]
}


@dataclass
class Rec:
    name: str
    title: str
    kind: str
    runnable: bool
    limit: int = 0
    est_minutes: float = 0.0
    why: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def model_bytes(params_b: float, quant: str) -> float:
    return params_b * 1e9 * (NF4_BYTES_PER_PARAM if quant == "4bit" else 2.0 if quant == "fp16" else 1.0)


def fits_resident(h: Hierarchy, params_b: float, quant: str, overhead_gb: float = 1.0) -> bool:
    """Can the model sit on the GPU (for in-process loglikelihood scoring)?"""
    return bool(h.vram) and model_bytes(params_b, quant) / GB + overhead_gb <= h.vram.capacity_gb * 0.9


def estimate_decode_tps(h: Hierarchy, params_b: float, quant: str) -> tuple[float, str]:
    """Tokens/s from memory bandwidth: each token reads every weight once, from VRAM or (offloaded part) RAM.

    Returns (tok/s, how it was derived). Checked once against llama.cpp: 8B Q4_K_M with 24/32 layers on a 4 GB
    GPU measured 10-18 tok/s; this model predicts ~10.
    """
    wbytes = model_bytes(params_b, quant)
    ram_bw = (h.ram.bandwidth_gbps or 10.0) * 1e9
    if not h.vram:
        return ram_bw / wbytes, "CPU only: RAM bandwidth / model size"
    assumed = not h.vram.bandwidth_gbps
    vram_bw = (h.vram.bandwidth_gbps or ASSUMED_VRAM_BW_GBPS) * 1e9
    usable = max(0.0, h.vram.capacity_gb * GB * 0.9 - 0.5 * GB)  # leave room for KV cache and compute buffers
    gpu_frac = min(1.0, usable / wbytes)
    per_token = gpu_frac * wbytes / vram_bw + (1 - gpu_frac) * wbytes / ram_bw
    how = "all on GPU" if gpu_frac >= 1 else f"{gpu_frac:.0%} of weights on GPU, rest streamed from RAM"
    note = (
        f"GPU bandwidth ASSUMED {ASSUMED_VRAM_BW_GBPS:g} GB/s (install PyTorch to measure)"
        if assumed
        else "bandwidth model"
    )
    return 1.0 / per_token, f"{how}: {note}"


def item_seconds(b: Benchmark, decode_tps: float) -> float:
    return (
        (b.prefill_tokens / (decode_tps * PREFILL_MULT))
        + (b.decode_tokens / decode_tps)
        + OVERHEAD_S_PER_ITEM
    )


def recommend(
    h: Hierarchy,
    params_b: float,
    quant: str = "4bit",
    minutes: float = 30.0,
    goal: str = "general",
    measured_tps: float | None = None,
    server_logprobs: bool = False,
) -> tuple[list[Rec], dict]:
    """Ranked plan: benchmarks relevant to `goal` first, each with the largest sample fitting its share of the
    time budget. Unrunnable ones are listed with the reason instead of being hidden."""
    tps, how = (
        (measured_tps, "measured (user supplied)")
        if measured_tps
        else estimate_decode_tps(h, params_b, quant)
    )
    resident = fits_resident(h, params_b, quant)
    relevant = [b for b in CATALOG.values() if goal in b.goals] or list(CATALOG.values())
    others = [b for b in CATALOG.values() if b not in relevant]
    runnable_rel = [b for b in relevant if _blocker(b, resident, server_logprobs) is None]
    share = minutes * 60 / max(1, len(runnable_rel))
    recs: list[Rec] = []
    for b in relevant + others:
        r = Rec(b.name, b.title, b.kind, True)
        blocker = _blocker(b, resident, server_logprobs)
        if blocker:
            r.runnable = False
            r.why.append(blocker)
        else:
            sec = item_seconds(b, tps)
            n = int(min(b.items, max(10, share // sec))) if b in relevant else min(b.items, 50)
            r.limit, r.est_minutes = n, round(n * sec / 60, 1)
            r.why.append(f"{b.measures}; ~{sec:.1f}s per item at {tps:.0f} tok/s -> {n} of {b.items} items")
            if b.note:
                r.why.append(b.note)
            if n < min(b.items, 100):
                r.why.append("time budget allows only a very small sample: expect large noise")
        if b in others:
            r.why.append(f"not a '{goal}' benchmark; shown for completeness")
        recs.append(r)
    ctx = {
        "decode_tps": round(tps, 1),
        "decode_tps_basis": how,
        "model_fits_gpu": resident,
        "budget_minutes": minutes,
        "goal": goal,
        "prefill_assumption": f"{PREFILL_MULT:.0f}x decode",
    }
    return recs, ctx


def _blocker(b: Benchmark, resident: bool, server_logprobs: bool) -> str | None:
    if b.kind == "loglik" and not resident and not server_logprobs:
        return (
            "loglikelihood scoring needs the model resident on the GPU (or a server that returns logprobs); "
            "this model does not fit"
        )
    return None


def lm_eval_cmd(
    b: Benchmark,
    limit: int,
    *,
    server_url: str = "",
    model_name: str = "default",
    hf_model: str = "",
    peft: str = "",
    out_dir: str = "bench_out",
    seed: int = 1234,
) -> list[str]:
    """`lm-evaluation-harness` command for one benchmark. Server targets use the chat-completions API."""
    if b.task.startswith("builtin:"):
        raise ValueError(f"{b.name} is a built-in benchmark, not an lm-eval task")
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--tasks",
        b.task,
        "--limit",
        str(limit),
        "--output_path",
        out_dir,
        "--seed",
        str(seed),
    ]
    if server_url:
        if b.kind == "loglik":
            cmd += [
                "--model",
                "local-completions",
                "--model_args",
                f"model={model_name},base_url={server_url.rstrip('/')}/v1/completions,num_concurrent=1,"
                "max_retries=2,tokenized_requests=False",
            ]
        else:
            cmd += [
                "--model",
                "local-chat-completions",
                "--apply_chat_template",
                "--model_args",
                f"model={model_name},base_url={server_url.rstrip('/')}/v1/chat/completions,num_concurrent=1,"
                "max_retries=2",
            ]
    elif hf_model:
        args = f"pretrained={hf_model},load_in_4bit=True"
        if peft:
            args += f",peft={peft}"
        cmd += ["--model", "hf", "--model_args", args, "--batch_size", "4"]
    else:
        raise ValueError("give either server_url or hf_model")
    return cmd


def render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def latest_results(out_dir: str) -> dict:
    """Metrics from the newest lm-eval `results_*.json` under out_dir ({} if none)."""
    import json
    from pathlib import Path

    files = sorted(Path(out_dir).rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {}
    try:
        return json.loads(files[-1].read_text(encoding="utf-8")).get("results", {})
    except (OSError, ValueError):
        return {}


def run_lm_eval(cmd: list[str], out_dir: str, runner=None) -> dict:
    """Run one harness command. `lm-eval` is optional: a missing install is reported, not raised."""
    import subprocess

    runner = runner or subprocess.run
    p = runner(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (p.stderr or p.stdout or "").strip().splitlines()[-6:]
    missing = "No module named lm_eval" in (p.stderr or "")
    return {
        "exit": p.returncode,
        "results": latest_results(out_dir) if p.returncode == 0 else {},
        "tail": tail,
        "missing_dependency": "lm-eval (pip install lm-eval)" if missing else None,
    }
