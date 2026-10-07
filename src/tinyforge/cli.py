"""`tinyforge` command line. Every command supports --json for machine-readable output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .worker import worker_app

app = typer.Typer(help="Train, evaluate and serve small LLMs on modest GPUs.", no_args_is_help=True,
                  pretty_exceptions_enable=False)  # main() turns missing-dependency errors into advice
data_app = typer.Typer(help="Dataset commands.")
app.add_typer(data_app, name="data")
app.add_typer(worker_app, name="worker")
console = Console()

JsonOpt = Annotated[bool, typer.Option("--json", help="Emit JSON lines.")]
STYLE = {"info": "cyan", "warn": "yellow", "error": "bold red"}


def _show(diags: list[dict]) -> None:
    for d in diags:
        fix = f"  -> {d['fix']}" if d.get("fix") else ""
        console.print(f"[{STYLE[d['level']]}]{d['level'].upper():5}[/] {d['code']}: "
                      f"{escape(d['message'] + fix)}")


@app.callback(invoke_without_command=True)
def _main(version: Annotated[bool, typer.Option("--version")] = False) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command()
def doctor(as_json: JsonOpt = False) -> None:
    """Probe hardware and report problems with fixes."""
    from . import deps, hardware

    hw = hardware.probe()
    rep = hardware.diagnose(hw)
    comps = deps.check_components()
    if as_json:
        print(json.dumps({"hardware": hw.to_dict(), "diagnostics": rep.to_list(), "components": comps}))
        return
    t = Table(title="Hardware")
    for k, v in hw.to_dict().items():
        t.add_row(k, str(v))
    console.print(t)
    ct = Table(title="Optional components")
    for col in ("component", "status", "unlocks", "how to get it"):
        ct.add_column(col)
    for comp in comps:
        status = "[green]ok[/]" if comp["installed"] else "[yellow]missing[/]"
        how = "" if comp["installed"] else escape(comp["install"])
        ct.add_row(comp["name"], status, comp["unlocks"], how)
    console.print(ct)
    _show(rep.to_list())
    raise typer.Exit(1 if rep.has_errors else 0)


@app.command()
def memory(
    params_b: Annotated[float, typer.Option("--params-b", help="Model size, billions of params.")] = 8.0,
    tokens: Annotated[int, typer.Option(help="Tokens per micro-batch.")] = 1024,
    quant: Annotated[str, typer.Option(help="4bit or none (fp16).")] = "4bit",
    no_measure: Annotated[bool, typer.Option("--no-measure", help="Skip bandwidth probes.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Probe VRAM/RAM/disk and say where a model's frozen weights would live and the estimated speed."""
    from . import memtiers

    h = memtiers.probe_hierarchy(measure=not no_measure)
    p = memtiers.plan_weights(h, params_b * 1e9, tokens, quant)
    warns = memtiers.ram_pressure_warnings(h)
    if as_json:
        from dataclasses import asdict

        print(json.dumps({"hierarchy": h.to_dict(), "placement": asdict(p), "warnings": warns}))
        return
    t = Table(title="Memory hierarchy")
    for c in ("tier", "kind", "capacity GB", "free GB", "GB/s", "measured"):
        t.add_column(c)
    for tier in (h.vram, h.ram, h.disk):
        if tier:
            bw = "-" if tier.bandwidth_gbps is None else f"{tier.bandwidth_gbps:.1f}"
            t.add_row(tier.name, tier.kind, f"{tier.capacity_gb}", f"{tier.free_gb}", bw, str(tier.measured))
    console.print(t)
    eta = f"{p.tokens_per_s:.0f} tok/s ({p.bound_by}-bound)" if p.tokens_per_s else "n/a"
    console.print(f"[bold]{params_b:g}B {quant}[/]: weights {p.weights_gb:.1f} GB -> {p.weights_tier}"
                  f"{' (streamed)' if p.stream else ''}, est. {eta}  [dim](estimate; MFU assumed)[/]")
    for r in p.reasons:
        console.print(f"  - {escape(r)}")
    for w in warns:
        console.print(f"[yellow]WARN[/] {escape(w)}")
    raise typer.Exit(0 if p.feasible else 1)


@data_app.command("prepare")
def data_prepare(
    source: Annotated[Path | None, typer.Option(help="Text file; omit to download TinyShakespeare.")] = None,
    out: Path = Path("data/tinyshakespeare"),
    vocab_size: int = 2048,
    as_json: JsonOpt = False,
) -> None:
    """Fetch/ingest text, train a BPE tokenizer, write train/val shards."""
    from . import data

    src = source or data.fetch_tinyshakespeare(out)
    meta, rep = data.prepare(src, out, vocab_size)
    if as_json:
        print(json.dumps({"meta": meta, "diagnostics": rep.to_list()}))
    else:
        console.print(meta)
        _show(rep.to_list())
    raise typer.Exit(1 if rep.has_errors else 0)


@data_app.command("ingest")
def data_ingest(
    paths: Annotated[list[Path], typer.Argument(help="Files/folders: txt, md, html, pdf*.")],
    out: Path = Path("data/docs.jsonl"),
    chunk_words: int = 300,
    min_words: int = 20,
    dedupe_threshold: float = 0.8,
    pii: Annotated[str, typer.Option(help="flag | redact | drop (secrets are always dropped)")] = "flag",
    as_json: JsonOpt = False,
) -> None:
    """Documents -> cleaned, chunked, de-duplicated text JSONL (step 1 of making training data)."""
    import dataclasses

    from . import etl

    chunks, meta, rep = etl.ingest(paths, chunk_words, min_words, dedupe_threshold, pii)
    if chunks:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(dataclasses.asdict(c), ensure_ascii=False) + "\n")
        meta["out"] = str(out)
    if as_json:
        print(json.dumps({"meta": meta, "diagnostics": rep.to_list()}))
    else:
        console.print(meta)
        _show(rep.to_list())
    raise typer.Exit(1 if rep.has_errors else 0)


@data_app.command("pairs")
def data_pairs(
    chunks: Path,
    out: Path = Path("data/pairs"),
    teacher_url: Annotated[str, typer.Option(help="OpenAI-compatible base URL (.../v1)")] = "",
    teacher_model: str = "default",
    api_key: str = "",
    per_chunk: int = 3,
    val_pct: int = 10,
    min_grounded: float = 0.8,
    as_json: JsonOpt = False,
) -> None:
    """Chunks -> grounded Q&A pairs. Splits by source file BEFORE the teacher runs; drops ungrounded."""
    from . import pairs

    if not teacher_url:
        console.print("[bold red]ERROR[/] --teacher-url is required (any OpenAI-compatible chat endpoint)")
        raise typer.Exit(1)
    teacher = pairs.openai_teacher(teacher_url, teacher_model, api_key)
    meta = pairs.run(chunks, out, teacher, f"{teacher_model}@{teacher_url}", per_chunk, val_pct, min_grounded)
    if as_json:
        print(json.dumps({"meta": meta}))
    else:
        console.print({k: v for k, v in meta.items() if k != "prompt"})


@data_app.command("pii-scan")
def data_pii_scan(
    path: Path,
    redact_to: Annotated[Path | None, typer.Option(help="Write a redacted copy here.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Scan a JSONL (messages or text rows) for PII and secrets. Exit 1 if any secret is found."""
    from . import pii

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def texts(r: dict) -> list[str]:
        if "messages" in r:
            return [m.get("content") or "" for m in r["messages"]]
        return [str(r.get("text", ""))]

    counts, hit, examples = pii.summarize(" \n".join(texts(r)) for r in rows)
    secrets = sum(n for k, n in counts.items() if k in pii.SECRET_PATTERNS)
    if redact_to is not None:
        with redact_to.open("w", encoding="utf-8") as f:
            for r in rows:
                if "messages" in r:
                    r = {**r, "messages": [{**m, "content": pii.redact(m.get("content") or "")}
                                           for m in r["messages"]]}
                else:
                    r = {**r, "text": pii.redact(str(r.get("text", "")))}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"rows": len(rows), "rows_with_findings": hit, "by_kind": dict(counts),
              "masked_examples": examples, "secrets": secrets,
              "note": "pattern-based: names, addresses, free-text identifiers are not detected"}
    if as_json:
        print(json.dumps(report))
    else:
        console.print(report)
    raise typer.Exit(1 if secrets else 0)


@data_app.command("tabular")
def data_tabular(
    csv_path: Path,
    out: Path = Path("data/tabular.jsonl"),
    count: int = 500,
    seed: int = 0,
    hard: Annotated[bool, typer.Option("--hard", help="Harder shapes (eval only).")] = False,
    as_json: JsonOpt = False,
) -> None:
    """CSV -> text-to-SQL examples whose answers were verified by executing them."""
    from . import tabular

    t = tabular.load_csv(csv_path)
    ex, meta = (tabular.generate_hard if hard else tabular.generate)(t, count, seed)
    if not ex:
        console.print("[bold red]ERROR[/] no verifiable examples (need a category or number column)")
        raise typer.Exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    meta["out"] = str(out)
    if as_json:
        print(json.dumps({"meta": meta}))
    else:
        console.print(meta)


@app.command()
def plan(
    preset: str = "micro", block_size: int = 256, batch_size: int = 16, grad_accum: int = 2,
    data_dir: Path = Path("data/tinyshakespeare"),
) -> None:
    """Show whether a preset fits your GPU *before* you start training."""
    from . import config, hardware

    hw = hardware.probe()
    mc = config.make_model_config(preset, _vocab(data_dir), block_size)
    tc = config.TrainConfig(batch_size=batch_size, grad_accum=grad_accum)
    _, rep = config.plan(mc, tc, hw.vram_gb, hw.bf16_supported)
    _show(rep.to_list())
    raise typer.Exit(1 if rep.has_errors else 0)


def _vocab(data_dir: Path) -> int:
    meta = data_dir / "meta.json"
    return json.loads(meta.read_text())["vocab_size"] if meta.exists() else 2048


@app.command()
def train(
    preset: str = "micro", steps: int = 2000, block_size: int = 256, batch_size: int = 16,
    grad_accum: int = 2, lr: float = 1e-3, dropout: float = 0.1, precision: str = "auto",
    compile: bool = False,
    data_dir: Path = Path("data/tinyshakespeare"), run_dir: Path = Path("runs/default"),
    as_json: JsonOpt = False,
) -> None:
    """Train a model from scratch (resumes automatically from run_dir/last.pt)."""
    from . import config
    from .train import train as run_train

    mc = config.make_model_config(preset, _vocab(data_dir), block_size)
    mc.dropout = dropout
    tc = config.TrainConfig(
        max_steps=steps, batch_size=batch_size, grad_accum=grad_accum, lr=lr, precision=precision,
        compile=compile, data_dir=data_dir, run_dir=run_dir,
        eval_interval=max(1, min(250, steps // 4)), ckpt_interval=max(1, min(500, steps // 2)),
        warmup_steps=min(100, max(1, steps // 10)))

    def on_event(e: dict) -> None:
        if as_json:
            print(json.dumps(e), flush=True)
        elif e["event"] == "step":
            console.print(f"step {e['step']:>6}  loss {e['loss']:.3f}  lr {e['lr']:.2e}  "
                          f"{e['tok_per_s']:,.0f} tok/s  mem {e['peak_mem_gb']:.2f} GB")
        elif e["event"] == "eval":
            console.print(f"[green]eval[/] step {e['step']}  val {e['val_loss']:.3f} "
                          f"(ppl {e['val_ppl']:.1f})  train {e['train_loss']:.3f}")
        elif e["event"] == "diagnostic":
            _show([e])

    try:
        run_train(mc, tc, on_event)
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(1) from exc


@app.command("eval")
def eval_(
    ckpt: Path = Path("runs/default/best.pt"), data_dir: Path = Path("data/tinyshakespeare"),
    out: Path | None = None, as_json: JsonOpt = False,
) -> None:
    """Run the evaluation suite (perplexity, quality, memorisation, correctness, speed)."""
    from .evaluate import evaluate

    res = evaluate(ckpt, data_dir, out or ckpt.parent / "eval.json")
    if as_json:
        print(json.dumps(res))
    else:
        for k in ("val_loss", "val_ppl", "train_loss", "distinct_2", "copy_rate_8gram",
                  "kv_cache_max_err", "decode_tok_per_s", "inference_peak_mem_gb"):
            console.print(f"{k:>24}: {res[k]:.4f}")
        _show(res["diagnostics"])
        console.print("[green]PASSED[/]" if res["passed"] else "[bold red]FAILED[/]")
    raise typer.Exit(0 if res["passed"] else 1)


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument()] = "ROMEO:", ckpt: Path = Path("runs/default/best.pt"),
    data_dir: Path = Path("data/tinyshakespeare"), max_new: int = 200, temperature: float = 0.8,
    top_k: int = 50, top_p: float = 0.95, int8: bool = False, triton: bool = False,
    seed: int | None = None,
) -> None:
    """Generate text from a checkpoint."""
    from . import hardware, infer
    from .data import load_tokenizer
    from .train import load_model

    model, _ = load_model(ckpt, hardware.probe().device)
    if int8:
        infer.quantize_int8(model)
    if triton and not infer.enable_triton():
        console.print("[yellow]Triton unavailable; using PyTorch kernels.[/]")
    console.print(infer.generate_text(model, load_tokenizer(data_dir), prompt, max_new=max_new,
                                      temperature=temperature, top_k=top_k, top_p=top_p, seed=seed))


@app.command()
def pipeline(
    preset: str = "micro", steps: int = 2000, workdir: Path = Path("."),
    vocab_size: int = 2048,
) -> None:
    """End to end: doctor -> data -> plan -> train -> eval. Exit code 1 if any gate fails."""
    import subprocess
    import sys

    base = [sys.executable, "-m", "tinyforge"]
    data_dir = workdir / "data" / "tinyshakespeare"
    run_dir = workdir / "runs" / preset
    stages = [
        ("doctor", base + ["doctor"]),
        ("data", base + ["data", "prepare", "--out", str(data_dir), "--vocab-size", str(vocab_size)]),
        ("plan", base + ["plan", "--preset", preset, "--data-dir", str(data_dir)]),
        ("train", base + ["train", "--preset", preset, "--steps", str(steps),
                          "--data-dir", str(data_dir), "--run-dir", str(run_dir)]),
        ("eval", base + ["eval", "--ckpt", str(run_dir / "best.pt"), "--data-dir", str(data_dir)]),
    ]
    for name, cmd in stages:
        console.rule(name)
        if subprocess.run(cmd).returncode != 0:
            console.print(f"[bold red]Stage '{name}' failed.[/]")
            raise typer.Exit(1)
    console.print("[bold green]Pipeline complete.[/]")


ft_app = typer.Typer(help="Fine-tune pretrained models with LoRA/QLoRA.")
app.add_typer(ft_app, name="ft")
BaseOpt = Annotated[str, typer.Option(help="Hugging Face model id.")]


def _ft_cfg(base_model: str, run_dir: Path, data_dir: Path, **kw):
    from .finetune import FTConfig

    return FTConfig(base_model=base_model, run_dir=run_dir, data_dir=data_dir, **kw)


@ft_app.command("data")
def ft_data(
    source: Annotated[str, typer.Option(help="JSONL path or HF dataset id.")] = "yahma/alpaca-cleaned",
    out: Path = Path("data/ft"), base_model: BaseOpt = "HuggingFaceTB/SmolLM2-360M-Instruct",
    limit: int = 3000, val_pct: int = 5, max_len: int = 512,
    pii: Annotated[str, typer.Option(help="flag | redact | drop (secrets are always dropped)")] = "flag",
    as_json: JsonOpt = False,
) -> None:
    """Normalise an instruction dataset into chat JSONL with dedupe, leakage-safe split, PII checks."""
    from . import ft_data as fd

    meta, rep = fd.prepare(source, out, base_model, limit, val_pct, max_len, pii)
    if as_json:
        print(json.dumps({"meta": meta, "diagnostics": rep.to_list()}))
    else:
        console.print(meta)
        _show(rep.to_list())
    raise typer.Exit(1 if rep.has_errors else 0)


@ft_app.command("plan")
def ft_plan(
    base_model: BaseOpt = "HuggingFaceTB/SmolLM2-360M-Instruct", max_len: int = 512,
    batch_size: int = 4, grad_accum: int = 4, quant: str = "auto", lora_r: int = 16,
) -> None:
    """Check whether fine-tuning fits your GPU, and how (4-bit? checkpointing? batch size?)."""
    from . import finetune, hardware

    c = _ft_cfg(base_model, Path("runs/ft"), Path("data/ft"), max_len=max_len, batch_size=batch_size,
                grad_accum=grad_accum, quant=quant, lora_r=lora_r)
    _, rep, _ = finetune.plan(c, hardware.probe().vram_gb)
    _show(rep.to_list())
    raise typer.Exit(1 if rep.has_errors else 0)


@ft_app.command("train")
def ft_train(
    base_model: BaseOpt = "HuggingFaceTB/SmolLM2-360M-Instruct", steps: int = 300,
    max_len: int = 512, batch_size: int = 4, grad_accum: int = 4, lr: float = 2e-4,
    lora_r: int = 16, quant: str = "auto", data_dir: Path = Path("data/ft"),
    run_dir: Path = Path("runs/ft"), as_json: JsonOpt = False,
) -> None:
    """LoRA fine-tune (resumes from run_dir/last.pt). Best adapter saved to run_dir/best."""
    from . import finetune

    c = _ft_cfg(base_model, run_dir, data_dir, max_steps=steps, max_len=max_len, batch_size=batch_size,
                grad_accum=grad_accum, lr=lr, lora_r=lora_r, lora_alpha=2 * lora_r, quant=quant,
                eval_interval=max(1, min(50, steps // 4)), warmup_steps=min(20, max(1, steps // 10)))

    def on_event(e: dict) -> None:
        if as_json:
            print(json.dumps(e), flush=True)
        elif e["event"] == "step":
            console.print(f"step {e['step']:>5}  loss {e['loss']:.3f}  lr {e['lr']:.2e}  "
                          f"{e['tok_per_s']:,.0f} tok/s  mem {e['peak_mem_gb']:.2f} GB")
        elif e["event"] == "eval":
            console.print(f"[green]eval[/] step {e['step']}  val {e['val_loss']:.3f} "
                          f"(ppl {e['val_ppl']:.2f})")
        elif e["event"] == "diagnostic":
            _show([e])

    try:
        finetune.train(c, on_event)
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(1) from exc


@ft_app.command("eval")
def ft_eval_cmd(
    run_dir: Path = Path("runs/ft"), data_dir: Path | None = None, merge: bool = True,
    as_json: JsonOpt = False,
) -> None:
    """Evaluate tuned vs base (held-out loss, forgetting, generations, merge check)."""
    from . import ft_eval

    res = ft_eval.evaluate(run_dir, data_dir, merge)
    if as_json:
        print(json.dumps(res))
    else:
        for k in ("val_loss_base", "val_loss_tuned", "improvement_pct", "general_loss_base",
                  "general_loss_tuned", "forgetting_pct", "distinct_2", "decode_tok_per_s",
                  "inference_peak_mem_gb"):
            console.print(f"{k:>24}: {res[k]:.4f}")
        for s in res["samples"][:2]:
            console.print(f"\n[bold]{s['prompt']}[/]\n[dim]base:[/]  {escape(s['base'][:300])}\n"
                          f"[dim]tuned:[/] {escape(s['tuned'][:300])}")
        _show(res["diagnostics"])
        console.print("[green]PASSED[/]" if res["passed"] else "[bold red]FAILED[/]")
    raise typer.Exit(0 if res["passed"] else 1)


@ft_app.command("task-eval")
def ft_task_eval(
    run_dir: Path = Path("runs/ft"), task: str = "sql", n: int = 100,
    min_gain_pts: Annotated[float, typer.Option(help="Required exact-match gain, in points.")] = 0.0,
    as_json: JsonOpt = False,
) -> None:
    """Score real task outputs (not just loss): tuned vs base on held-out examples."""
    from . import ft_task

    res = ft_task.evaluate_task(run_dir, task, n, min_gain_pts)
    if as_json:
        print(json.dumps(res))
    else:
        t = Table(title=f"{task}: {res['n']} held-out examples")
        t.add_column("metric")
        t.add_column("base (raw)")
        t.add_column("base (instructed)")
        t.add_column("tuned")
        for k in ("strict_em", "lenient_em", "valid", "exec_acc"):
            t.add_row(k, f"{res['base_raw_prompt'][k]:.1%}", f"{res['base_instructed'][k]:.1%}",
                      f"{res['tuned'][k]:.1%}")
        console.print(t)
        console.print(f"gain: {res['gain_pts_lenient_em']:+.1f} points exact-match")
        _show(res["diagnostics"])
        console.print("[green]PASSED[/]" if res["passed"] else "[bold red]FAILED[/]")
    raise typer.Exit(0 if res["passed"] else 1)


@ft_app.command("card")
def ft_card(run_dir: Path = Path("runs/ft"), out: Path | None = None) -> None:
    """Write a model card (README.md) from the run's own config and eval results."""
    from . import modelcard

    dest = out or run_dir / "MODEL_CARD.md"
    dest.write_text(modelcard.render(run_dir), encoding="utf-8")
    console.print(f"wrote {dest}")


@ft_app.command("generate")
def ft_generate(
    prompt: Annotated[str, typer.Argument()], run_dir: Path = Path("runs/ft"),
    max_new: int = 200, base: bool = False,
) -> None:
    """Chat with the tuned model (or --base for the original)."""
    from peft import PeftModel

    from . import hardware
    from .finetune import FTConfig, load_base, load_tokenizer
    from .ft_eval import _gen

    cfg = FTConfig.model_validate_json((run_dir / "ft_config.json").read_text())
    hw = hardware.probe()
    model = load_base(cfg.base_model, cfg.quant, hw.bf16_supported, hw.device)
    model = PeftModel.from_pretrained(model, run_dir / "best")
    tok = load_tokenizer(cfg.base_model)
    if base:
        with model.disable_adapter():
            console.print(escape(_gen(model, tok, prompt, hw.device, max_new)))
    else:
        console.print(escape(_gen(model, tok, prompt, hw.device, max_new)))


@ft_app.command("pipeline")
def ft_pipeline(
    base_model: BaseOpt = "HuggingFaceTB/SmolLM2-360M-Instruct", steps: int = 300,
    source: str = "yahma/alpaca-cleaned", limit: int = 3000, workdir: Path = Path("."),
) -> None:
    """End to end: doctor -> data -> plan -> train -> eval/merge. Exit 1 if any gate fails."""
    import subprocess
    import sys

    base = [sys.executable, "-m", "tinyforge"]
    d, r = str(workdir / "data" / "ft"), str(workdir / "runs" / "ft")
    stages = [
        ("doctor", base + ["doctor"]),
        ("data", base + ["ft", "data", "--source", source, "--out", d, "--base-model", base_model,
                         "--limit", str(limit)]),
        ("plan", base + ["ft", "plan", "--base-model", base_model]),
        ("train", base + ["ft", "train", "--base-model", base_model, "--steps", str(steps),
                          "--data-dir", d, "--run-dir", r]),
        ("eval", base + ["ft", "eval", "--run-dir", r]),
    ]
    for name, cmd in stages:
        console.rule(name)
        if subprocess.run(cmd).returncode != 0:
            console.print(f"[bold red]Stage '{name}' failed.[/]")
            raise typer.Exit(1)
    console.print("[bold green]Fine-tuning pipeline complete.[/]")


methods_app = typer.Typer(help="Choose a fine-tuning method that fits this machine.")
app.add_typer(methods_app, name="methods")


@methods_app.command("list")
def methods_list(as_json: JsonOpt = False) -> None:
    """The fine-tuning methods the tool knows, and what has actually been verified."""
    from dataclasses import asdict

    from . import methods

    if as_json:
        print(json.dumps([asdict(m) for m in methods.CATALOG.values()]))
        return
    t = Table(title="Fine-tuning methods")
    for c in ("name", "family", "data", "backends", "verified here"):
        t.add_column(c)
    for m in methods.CATALOG.values():
        t.add_row(m.name, m.family, m.data, ",".join(m.backends) or "-", m.verified)
    console.print(t)


@methods_app.command("suggest")
def methods_suggest(
    params_b: Annotated[float, typer.Option("--params-b", help="Model size, billions of params.")] = 8.0,
    data: Annotated[str, typer.Option(help="sft (prompt/answer) | preference (chosen/rejected)")] = "sft",
    prefer: Annotated[str, typer.Option(help="quality | fit")] = "quality",
    no_measure: Annotated[bool, typer.Option("--no-measure", help="Skip bandwidth probes.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Which methods fit this GPU/RAM for a model of this size, on which backend, and why."""
    from . import deps, memtiers, methods

    h = memtiers.probe_hierarchy(measure=not no_measure)
    have = {c["name"] for c in deps.check_components() if c["installed"]}
    backends = tuple(b for b, need in (("native", "finetune"), ("soup", "soup")) if need in have)
    recs, ctx = methods.recommend_methods(h, params_b, data, backends or ("native", "soup"), prefer)
    ctx["installed_backends"] = list(backends)
    if as_json:
        print(json.dumps({"context": ctx, "methods": [r.to_dict() for r in recs]}))
        return
    have_txt = ", ".join(backends) or "none (showing what would fit)"
    hw_txt = f"{ctx['vram_gb']:g} GB VRAM, {ctx['free_ram_gb']} GB RAM free"
    console.print(f"[bold]{params_b:g}B model[/]: {hw_txt}; backends installed: {have_txt}")
    t = Table()
    for c in ("#", "method", "fits", "backend", "~GB", "why"):
        t.add_column(c)
    for r in recs:
        t.add_row(str(r.rank or "-"), r.name, "yes" if r.fits else "no", r.backend, str(r.need_gb),
                  escape(" | ".join(r.why[1:] or r.why)))
    console.print(t)


bench_app = typer.Typer(help="Pick and run standard benchmarks sized to this machine.")
app.add_typer(bench_app, name="bench")


@bench_app.command("list")
def bench_list(as_json: JsonOpt = False) -> None:
    """The benchmark catalog: what each measures and how it is run."""
    from . import bench

    if as_json:
        from dataclasses import asdict

        print(json.dumps([asdict(b) for b in bench.CATALOG.values()]))
        return
    t = Table(title="Benchmarks")
    for c in ("name", "kind", "items", "measures", "goals"):
        t.add_column(c)
    for b in bench.CATALOG.values():
        t.add_row(b.name, b.kind, str(b.items), b.measures, ",".join(b.goals))
    console.print(t)


@bench_app.command("suggest")
def bench_suggest(
    params_b: Annotated[float, typer.Option("--params-b", help="Model size, billions of params.")] = 8.0,
    quant: Annotated[str, typer.Option(help="4bit | fp16")] = "4bit",
    minutes: Annotated[float, typer.Option(help="Time budget for the whole suite.")] = 30.0,
    goal: Annotated[str, typer.Option(help="general|forgetting|reasoning|instruction|sql")] = "general",
    measured_tps: Annotated[float, typer.Option(help="Measured decode tok/s (overrides estimate).")] = 0.0,
    server_logprobs: Annotated[bool, typer.Option(help="Server returns logprobs (loglik tasks).")] = False,
    no_measure: Annotated[bool, typer.Option("--no-measure", help="Skip bandwidth probes.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Recommend benchmarks and sample sizes for this GPU/RAM, model size and time budget."""
    from . import bench, memtiers

    h = memtiers.probe_hierarchy(measure=not no_measure)
    recs, ctx = bench.recommend(h, params_b, quant, minutes, goal, measured_tps or None, server_logprobs)
    if as_json:
        print(json.dumps({"context": ctx, "recommendations": [r.to_dict() for r in recs]}))
        return
    console.print(f"[bold]{params_b:g}B {quant}[/] on this machine: ~{ctx['decode_tps']} tok/s "
                  f"({escape(ctx['decode_tps_basis'])}); model fits GPU: {ctx['model_fits_gpu']}; "
                  f"budget {minutes:g} min; goal {goal}  [dim](estimates)[/]")
    t = Table()
    for c in ("benchmark", "runnable", "items", "~min", "why"):
        t.add_column(c)
    for r in recs:
        t.add_row(r.name, "yes" if r.runnable else "no", str(r.limit or "-"),
                  f"{r.est_minutes:g}" if r.runnable else "-", escape(" | ".join(r.why)))
    console.print(t)


@bench_app.command("run")
def bench_run(
    names: Annotated[list[str], typer.Argument(help="Benchmark names (see `bench list`).")],
    server_url: Annotated[str, typer.Option(help="OpenAI-compatible server URL.")] = "",
    hf_model: Annotated[str, typer.Option(help="Local HF model folder (in-process, 4-bit).")] = "",
    peft: Annotated[str, typer.Option(help="LoRA adapter folder for --hf-model.")] = "",
    limit: Annotated[int, typer.Option(help="Items per benchmark (sub-sample).")] = 100,
    out: Path = Path("bench_out"),
    compare: Annotated[bool, typer.Option(help="llama-server + LoRA: run adapter on and off.")] = False,
    csv_path: Annotated[Path | None, typer.Option("--csv", help="sql-exec: the table CSV.")] = None,
    test_file: Annotated[Path | None, typer.Option(help="sql-exec: JSONL of verified questions.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the commands, run nothing.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Run benchmarks (lm-evaluation-harness, or the built-in SQL execution check) and write a summary."""
    import httpx

    from . import bench, sqleval

    unknown = [n for n in names if n not in bench.CATALOG]
    if unknown:
        console.print(f"[bold red]ERROR[/] unknown benchmark(s) {unknown}; see `tinyforge bench list`")
        raise typer.Exit(2)
    labels = [("tuned", 1.0), ("base", 0.0)] if (compare and server_url) else [("model", None)]
    summary: dict = {"limit": limit, "target": server_url or hf_model, "runs": {}}
    for name in names:
        b = bench.CATALOG[name]
        for label, scale in labels:
            key = f"{name}/{label}"
            if b.task == "builtin:sql-exec":
                if not (server_url and csv_path and test_file):
                    console.print("[bold red]ERROR[/] sql-exec needs --server-url, --csv and --test-file")
                    raise typer.Exit(2)
                if dry_run:
                    summary["runs"][key] = {"dry_run": f"sql-exec {server_url} n={limit}"}
                    break
                variants = ("tuned", "base_instructed") if compare else ("base_instructed",)
                summary["runs"][name] = sqleval.run(server_url, csv_path, test_file, limit, variants)
                break
            cmd = bench.lm_eval_cmd(b, limit, server_url=server_url, hf_model=hf_model, peft=peft,
                                    out_dir=str(out / key))
            if dry_run:
                summary["runs"][key] = {"dry_run": bench.render_cmd(cmd)}
                continue
            if scale is not None:
                httpx.post(f"{server_url.rstrip('/')}/lora-adapters", json=[{"id": 0, "scale": scale}],
                           timeout=60).raise_for_status()
            summary["runs"][key] = bench.run_lm_eval(cmd, str(out / key))
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if as_json:
        print(json.dumps(summary))
    else:
        console.print(summary)
    failed = [k for k, v in summary["runs"].items() if isinstance(v, dict) and v.get("exit", 0) != 0]
    raise typer.Exit(1 if failed else 0)


export_app = typer.Typer(help="Export models for other runtimes.")
app.add_typer(export_app, name="export")


@export_app.command("gguf")
def export_gguf_cmd(
    base: Annotated[Path, typer.Option(help="Local Hugging Face model folder (safetensors).")],
    out: Annotated[Path, typer.Option(help="Output folder.")] = Path("out/gguf"),
    adapter: Annotated[Path | None, typer.Option(help="LoRA adapter folder (e.g. runs/x/best).")] = None,
    quant: Annotated[str, typer.Option(help="f16 | q8_0 | q4_k_m | q5_k_m | q6_k | q4_0")] = "q4_k_m",
    keep_f16: Annotated[bool, typer.Option(help="Keep the large f16 intermediate.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Base model (+ optional LoRA adapter) -> GGUF files for llama.cpp (needs the llama.cpp tools)."""
    from .gguf_export import ExportError, export_gguf

    try:
        res = export_gguf(base, out, adapter, quant, keep_f16, log=lambda s: typer.echo(s, err=True))
    except ExportError as e:
        console.print(f"[bold red]ERROR[/] {escape(str(e))}")
        raise typer.Exit(1) from e
    if as_json:
        print(json.dumps(res))
    else:
        console.print(res)


@app.command()
def serve(
    host: str = "127.0.0.1", port: int = 8000,
    ssl_certfile: Annotated[Path | None, typer.Option(
        envvar="TINYFORGE_SSL_CERTFILE", help="PEM certificate (chain) to serve HTTPS (TLS 1.2+).")] = None,
    ssl_keyfile: Annotated[Path | None, typer.Option(
        envvar="TINYFORGE_SSL_KEYFILE", help="PEM private key for --ssl-certfile.")] = None,
    ssl_ca_certs: Annotated[Path | None, typer.Option(
        envvar="TINYFORGE_SSL_CA_CERTS", help="CA bundle used to verify client certificates (mTLS).")] = None,
    client_cert_required: Annotated[bool, typer.Option(
        "--client-cert-required",
        help="mTLS: reject clients without a certificate signed by --ssl-ca-certs.")] = False,
    insecure_http: Annotated[bool, typer.Option(
        "--insecure-http",
        help="Allow plain HTTP on a non-loopback address (trusted private network only).")] = False,
    engine: Annotated[str, typer.Option(help="hf (in-process) | llamacpp (GGUF via llama-server)")] = "",
    gguf: Annotated[Path | None, typer.Option(help="Base GGUF for --engine llamacpp.")] = None,
    lora_gguf: Annotated[Path | None, typer.Option(help="LoRA adapter GGUF for --engine llamacpp.")] = None,
) -> None:
    """Start the API + web UI.

    Binding anything but loopback needs TLS (or --insecure-http) AND TINYFORGE_API_TOKEN. In production
    terminate TLS at an ingress / load balancer; use --client-cert-required for east-west mTLS.
    """
    import uvicorn

    from .netsec import ServeConfigError, resolve_serve

    try:
        cfg = resolve_serve(host, port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
                            ssl_ca_certs=ssl_ca_certs, client_cert_required=client_cert_required,
                            insecure_http=insecure_http)
    except ServeConfigError as e:
        typer.echo(f"serve: {e}", err=True)
        raise typer.Exit(2) from e
    for w in cfg.warnings:
        typer.echo(f"WARNING: {w}", err=True)
    if engine:  # the server module reads these when it is imported by uvicorn below
        os.environ["TINYFORGE_ENGINE"] = engine
    if gguf:
        os.environ["TINYFORGE_GGUF"] = str(gguf)
    if lora_gguf:
        os.environ["TINYFORGE_LORA_GGUF"] = str(lora_gguf)
    uvicorn.run("tinyforge.server:app", host=cfg.host, port=cfg.port, **cfg.uvicorn)


def main() -> None:
    """Console entry point: a missing optional dependency gets advice and exit 2, not a traceback."""
    import sys

    from . import deps

    try:
        app()
    except ModuleNotFoundError as e:
        msg = deps.explain_missing(e)
        if msg is None:
            raise
        print(f"tinyforge: {msg}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
