"""`tinyforge` command line. Every command supports --json for machine-readable output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .worker import worker_app

app = typer.Typer(help="Train, evaluate and serve small LLMs on modest GPUs.", no_args_is_help=True)
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
    from . import hardware

    hw = hardware.probe()
    rep = hardware.diagnose(hw)
    if as_json:
        print(json.dumps({"hardware": hw.to_dict(), "diagnostics": rep.to_list()}))
        return
    t = Table(title="Hardware")
    for k, v in hw.to_dict().items():
        t.add_row(k, str(v))
    console.print(t)
    _show(rep.to_list())
    raise typer.Exit(1 if rep.has_errors else 0)


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
    limit: int = 3000, val_pct: int = 5, max_len: int = 512, as_json: JsonOpt = False,
) -> None:
    """Normalise an instruction dataset into chat JSONL with dedupe, leakage-safe split, PII checks."""
    from . import ft_data as fd

    meta, rep = fd.prepare(source, out, base_model, limit, val_pct, max_len)
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
        for k in ("strict_em", "lenient_em", "valid"):
            t.add_row(k, f"{res['base_raw_prompt'][k]:.1%}", f"{res['base_instructed'][k]:.1%}",
                      f"{res['tuned'][k]:.1%}")
        console.print(t)
        console.print(f"gain: {res['gain_pts_lenient_em']:+.1f} points exact-match")
        _show(res["diagnostics"])
        console.print("[green]PASSED[/]" if res["passed"] else "[bold red]FAILED[/]")
    raise typer.Exit(0 if res["passed"] else 1)


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
    uvicorn.run("tinyforge.server:app", host=cfg.host, port=cfg.port, **cfg.uvicorn)


if __name__ == "__main__":
    app()
