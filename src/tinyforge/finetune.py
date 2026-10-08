"""LoRA / QLoRA fine-tuning of pretrained causal LMs, planned to fit small GPUs.

Same event stream and warning system as scratch training, so CLI/API/UI treat both alike.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path

import torch
from pydantic import BaseModel

from . import deps, hardware
from .errors import UserError
from .memory import causal_lm_loss
from .warnings import Level, Report

deps.shield_old_torchao()  # peft refuses torchao < 0.16 (Colab/Kaggle ship 0.10); we never use it

DEFAULT_BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"


UNTRAINED_FRAC = 0.9  # held-out loss >= this fraction of ln(vocab) means 'no better than random guessing'


class FTConfig(BaseModel):
    base_model: str = DEFAULT_BASE
    run_dir: Path = Path("runs/ft")
    data_dir: Path = Path("data/ft")
    max_len: int = 512
    max_steps: int = 300
    batch_size: int = 4  # micro-batch examples at max_len (the planner may raise it)
    grad_accum: int = 4  # examples per optimizer step = batch_size * grad_accum
    examples_per_step: int = 0  # set by the planner; 0 -> batch_size * grad_accum
    token_budget: int = 0  # max padded tokens per micro-batch; 0 -> batch_size * max_len
    lr: float = 2e-4
    warmup_steps: int = 20
    grad_clip: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0  # 0 lets PEFT skip dropout kernels: measured ~40% faster
    quant: str = "auto"  # auto | none | 4bit
    grad_checkpointing: bool = False
    auto_plan: bool = True  # False: use batch/quant/checkpointing exactly as given (benchmarks, expert use)
    chunked_ce: bool = True  # project only labelled positions through the LM head, in chunks (see memory.py)
    ce_chunk: int = 2048
    eval_interval: int = 50
    eval_examples: int = 64
    seed: int = 1337


# ---------------------------------------------------------------- loading / planning


def _dtype(bf16: bool):
    return torch.bfloat16 if bf16 else torch.float16


def load_base(name: str, quant: str, bf16: bool, device: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dt = _dtype(bf16 and device == "cuda") if device == "cuda" else torch.float32
    if quant == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb,
                                                    device_map={"": torch.cuda.current_device()}, dtype=dt)
    return AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(device)


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def model_shape(name: str) -> dict:
    """Parameter counts and dims from a meta-device model: no weights are downloaded or allocated."""
    import torch.nn as nn
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(name)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    linear_dims = [mod.in_features + mod.out_features for n, mod in m.named_modules()
                   if isinstance(mod, nn.Linear) and "lm_head" not in n]
    return {"params": sum(p.numel() for p in m.parameters()), "hidden": cfg.hidden_size,
            "layers": cfg.num_hidden_layers, "vocab": cfg.vocab_size, "linear_dims": linear_dims}


def estimate_vram_gb(shape: dict, c: FTConfig, quant: str, checkpointing: bool, batch: int) -> float:
    weights = shape["params"] * (0.55 if quant == "4bit" else 2)
    lora = c.lora_r * sum(shape["linear_dims"]) * (4 + 4 + 8)  # fp32 weights + grads + Adam
    tokens = batch * c.max_len
    # Factors calibrated against measured peaks on SmolLM2-360M (RTX 3050 Ti), not theory:
    # without checkpointing every layer keeps ~80 bytes/token/hidden; with it, only layer inputs.
    if checkpointing:
        acts = shape["layers"] * tokens * shape["hidden"] * 2 + 34 * tokens * shape["hidden"] * 2
    else:
        acts = shape["layers"] * 80 * tokens * shape["hidden"]
    # bf16 logits + fp32 upcast + grad; chunked CE only ever holds one chunk (at most `tokens` of them)
    logits = (min(tokens, c.ce_chunk) if c.chunked_ce else tokens) * shape["vocab"] * 10
    return (weights + lora + acts + logits + 0.6 * 1024**3) / 1024**3


def plan(c: FTConfig, vram_gb: float, shape: dict | None = None) -> tuple[FTConfig, Report, dict]:
    """Pick (quantisation, checkpointing, micro-batch) that fits; preserve effective batch size."""
    r = Report()
    shape = shape or model_shape(c.base_model)
    t = c.model_copy()
    eff = c.batch_size * c.grad_accum
    r.add("FP001", Level.INFO, f"{c.base_model}: {shape['params'] / 1e6:.0f}M params, "
          f"{shape['layers']} layers, vocab {shape['vocab']:,}.")
    if vram_gb <= 0:
        r.add("FP002", Level.WARN, "No GPU: fine-tuning on CPU works only for toy runs.")
        t.quant = "none"
        return t, r, shape
    budget = vram_gb * 0.80  # leave room for CUDA context, cuBLAS workspaces and fragmentation
    quants = ["none", "4bit"] if c.quant == "auto" else [c.quant]
    cks = [True] if c.grad_checkpointing else [False, True]
    # Throughput-first: on small GPUs training is launch-bound, so throughput scales ~linearly
    # (measured: bs8 = 2.3x bs4) with micro-batch size. Checkpointing costs ~1.5x compute but buys a
    # much larger batch, so pick the (checkpointing, batch) pair with the best bs / cost score.
    # Prefer unquantised weights; only fall to 4-bit when nothing else fits.
    for q in quants:
        cands = []
        for ck in cks:
            for bs in range(min(eff, 32), 0, -1):  # largest worst-case (max_len) batch that fits
                need = estimate_vram_gb(shape, c, q, ck, bs)
                if need <= budget:
                    cands.append((bs / (1.5 if ck else 1.0), ck, bs, need))
                    break
        if not cands:
            continue
        _, ck, bs, need = max(cands)
        t.quant, t.grad_checkpointing, t.batch_size = q, ck, bs
        t.examples_per_step = eff
        t.token_budget = bs * c.max_len
        t.grad_accum = max(1, math.ceil(eff / bs))
        if q == "4bit" and c.quant == "auto":
            r.add("FP003", Level.WARN, "fp16/bf16 weights do not fit; switched to 4-bit "
                  "QLoRA (NF4). Slightly lower quality, ~4x less weight memory.")
        if ck and not c.grad_checkpointing:
            r.add("FP004", Level.INFO, "Enabled gradient checkpointing (~1.5x compute, but a much "
                  "larger micro-batch, which is faster overall on small GPUs).")
        r.add("FP006", Level.INFO, f"Plan: quant={q}, checkpointing={ck}, up to {bs} examples "
              f"({t.token_budget:,} tokens) per micro-batch, {eff} examples per optimizer step; "
              f"est. worst case {need:.1f} GB of {vram_gb:.1f} GB.")
        return t, r, shape
    r.add("FP007", Level.ERROR, f"No configuration fits in {vram_gb:.1f} GB at max_len={c.max_len}.",
          "Pick a smaller base model or lower --max-len.")
    return t, r, shape


# ---------------------------------------------------------------- data


def turn_terminator(tok) -> str:
    """The token string the chat template puts after an assistant reply (falls back to eos_token).

    Base models often have eos_token=<|endoftext|> while their template ends turns with <|im_end|>;
    training on the wrong one makes template-aware servers (llama.cpp, vLLM chat) run past the reply.
    """
    probe = "@@reply@@"
    try:
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": probe}]
        full = tok.apply_chat_template(convo, tokenize=False)
    except Exception:
        return tok.eos_token
    tail = full.split(probe, 1)[1].strip() if probe in full else ""
    return tail if tail and tail in tok.get_vocab() else tok.eos_token


def require_chat_template(tok, base_model: str = "") -> None:
    """Fine-tuning here is chat-format SFT: the tokenizer must define how to render a conversation."""
    if not getattr(tok, "chat_template", None):
        raise UserError(
            f"The tokenizer of '{base_model or 'this model'}' has no chat template, so tinyforge cannot "
            "format conversations for it. Use an instruct/chat version of the model (for example a "
            "'-Instruct' checkpoint), or set tokenizer.chat_template yourself.")


class ChatDataset:
    """Tokenised chat examples with loss masked to the assistant reply only."""

    def __init__(self, path: Path, tok, max_len: int):
        require_chat_template(tok)
        self.items: list[tuple[list[int], list[int]]] = []
        self.truncated = 0
        end = turn_terminator(tok)
        for line in path.read_text(encoding="utf-8").splitlines():
            msgs = json.loads(line)["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            a_ids = tok(msgs[-1]["content"] + end, add_special_tokens=False).input_ids
            ids = p_ids + a_ids
            labels = [-100] * len(p_ids) + a_ids
            if len(ids) > max_len:
                self.truncated += 1
                ids, labels = ids[:max_len], labels[:max_len]
            if any(label != -100 for label in labels):  # skip if the reply was cut off entirely
                self.items.append((ids, labels))
        self.pad = tok.pad_token_id

    def __len__(self) -> int:
        return len(self.items)

    def batches(self, token_budget: int, rng: random.Random, max_n: int = 32) -> list[list[int]]:
        """Token-budget batches: sort by length inside random mega-chunks (little padding), then pack
        as many examples as fit in `token_budget` padded tokens. Short examples => big batches."""
        idx = list(range(len(self.items)))
        rng.shuffle(idx)
        out: list[list[int]] = []
        for i in range(0, len(idx), 256):
            cur: list[int] = []
            for j in sorted(idx[i:i + 256], key=lambda k: len(self.items[k][0])):
                longest = len(self.items[j][0])  # ascending, so this is the batch max
                if cur and ((len(cur) + 1) * longest > token_budget or len(cur) >= max_n):
                    out.append(cur)
                    cur = []
                cur.append(j)
            if cur:
                out.append(cur)
        rng.shuffle(out)
        return out

    def collate(self, idx: list[int], device: str) -> dict:
        batch = [self.items[i] for i in idx]
        n = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), n), self.pad, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)], lab[i, :len(y)], att[i, :len(x)] = (
                torch.tensor(x), torch.tensor(y), 1)
        return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}


@torch.no_grad()
def eval_loss(model, ds: ChatDataset, n: int, bs: int, device: str, amp, ce_chunk: int = 0) -> float:
    """Token-weighted mean loss over the first n examples (fixed set => comparable across runs)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    idx = list(range(min(n, len(ds))))
    for i in range(0, len(idx), bs):
        b = ds.collate(idx[i:i + bs], device)
        with torch.autocast(device, dtype=amp, enabled=amp is not None):
            loss = causal_lm_loss(model, b, ce_chunk) if ce_chunk else model(**b).loss
        ntok = (b["labels"][:, 1:] != -100).sum().item()
        total += loss.item() * ntok
        count += ntok
    model.train(was_training)
    return total / max(1, count)


# ---------------------------------------------------------------- training


def dist_setup() -> tuple[int, int]:
    """(rank, world_size). Data parallel is enabled by launching under torchrun (WORLD_SIZE > 1); each process
    owns one GPU (LOCAL_RANK) and the LoRA gradients are averaged across them every optimizer step."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1
    import torch.distributed as dist

    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
        init = os.environ.get("TINYFORGE_DIST_INIT")  # file rendezvous (Windows, where torchrun cannot start)
        if init:
            dist.init_process_group(backend, init_method=init, rank=rank, world_size=world)
        else:
            dist.init_process_group(backend)
    return rank, world


def _allreduce_mean_grads(params: list, world: int) -> None:
    """Average .grad across ranks in one flat all-reduce per dtype (LoRA grads are small)."""
    import torch.distributed as dist

    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    by_dtype: dict = {}
    for p in params:
        by_dtype.setdefault(p.grad.dtype, []).append(p)
    for ps in by_dtype.values():
        flat = torch._utils._flatten_dense_tensors([p.grad for p in ps])
        dist.all_reduce(flat)
        flat /= world
        for p, g in zip(ps, torch._utils._unflatten_dense_tensors(flat, [p.grad for p in ps]), strict=True):
            p.grad.copy_(g)


def _scalar(value: float, device: str):
    return torch.tensor([value], dtype=torch.float64, device=device if device == "cuda" else "cpu")


def _dist_max(value: float, device: str) -> float:
    import torch.distributed as dist

    t = _scalar(value, device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _dist_sum(value: float, device: str) -> float:
    import torch.distributed as dist

    t = _scalar(value, device)
    dist.all_reduce(t)
    return t.item()


def _is_mem_error(exc: BaseException) -> bool:
    s = str(exc)
    return (isinstance(exc, torch.OutOfMemoryError) or "out of memory" in s.lower()
            or "CUBLAS_STATUS_INTERNAL_ERROR" in s or "CUBLAS_STATUS_ALLOC_FAILED" in s)


def _lr(step: int, c: FTConfig) -> float:
    if step < c.warmup_steps:
        return c.lr * (step + 1) / c.warmup_steps
    prog = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
    return c.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def train(c: FTConfig, on_event: Callable[[dict], None] | None = None) -> dict:
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    rank, world = dist_setup()
    lead = rank == 0  # only the lead rank logs, writes files and reports events
    emit = on_event or (lambda e: None)
    c.run_dir.mkdir(parents=True, exist_ok=True)
    log = open(c.run_dir / "metrics.jsonl", "a", buffering=1) if lead else open(os.devnull, "w")

    def event(kind: str, **kw):
        if not lead:
            return
        e = {"event": kind, "t": time.time(), **kw}
        log.write(json.dumps(e) + "\n")
        emit(e)

    hw = hardware.probe()
    if c.auto_plan:
        c, rep, _ = plan(c, hw.vram_gb)
    else:
        rep = Report()
        if c.quant == "auto":
            c = c.model_copy(update={"quant": "none"})
    for d in rep.to_list():
        event("diagnostic", **d)
    if rep.has_errors:
        event("failed", reason="plan did not fit hardware")
        raise RuntimeError("Fine-tuning plan does not fit available hardware; see diagnostics.")
    if lead:
        (c.run_dir / "ft_config.json").write_text(c.model_dump_json(indent=2))

    device = hw.device
    random.seed(c.seed)
    torch.manual_seed(c.seed)
    amp = (torch.bfloat16 if hw.bf16_supported else torch.float16) if device == "cuda" else None

    tok = load_tokenizer(c.base_model)
    tr = ChatDataset(c.data_dir / "train.jsonl", tok, c.max_len)
    va = ChatDataset(c.data_dir / "val.jsonl", tok, c.max_len)
    if tr.truncated:
        event("diagnostic", code="FT004", level="info", fix="",
              message=f"{tr.truncated}/{len(tr) + tr.truncated} training examples truncated to {c.max_len}.")

    model = load_base(c.base_model, c.quant, hw.bf16_supported, device)
    if c.quant == "4bit":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=c.grad_checkpointing)
    elif c.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if world > 1:  # the same seed already gives the same init; broadcasting makes it certain
        import torch.distributed as dist

        for p in trainable:
            dist.broadcast(p.data, 0)
    n_train = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=c.lr, weight_decay=0.0, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(device, enabled=amp is torch.float16)

    step, best_val = 0, float("inf")
    last = c.run_dir / "last.pt"
    if last.exists():
        blob = torch.load(last, map_location=device, weights_only=True)
        set_peft_model_state_dict(model, blob["adapter"])
        opt.load_state_dict(blob["opt"])
        step, best_val = blob["step"], blob["best_val"]
        event("resumed", step=step)

    event("started", params=n_train, total_params=sum(p.numel() for p in model.parameters()),
          device=device, precision=str(amp), quant=c.quant, max_steps=c.max_steps,
          train_examples=len(tr), val_examples=len(va), config=c.model_dump(mode="json"))

    eps_global = c.examples_per_step or c.batch_size * c.grad_accum
    eps = max(1, eps_global // world)  # per-rank share; the global batch stays ~eps_global
    budget_tokens = c.token_budget or c.batch_size * c.max_len
    ce = c.ce_chunk if c.chunked_ce else 0
    eval_bs = min(c.batch_size, 4)
    model.train()
    if step == 0:
        base_val = eval_loss(model, va, c.eval_examples, eval_bs, device, amp, ce)
        best_val = base_val
        event("eval", step=0, val_loss=base_val, train_loss=None,
              val_ppl=math.exp(min(base_val, 20)))
        rand_loss = math.log(max(2, len(tok)))
        if base_val >= UNTRAINED_FRAC * rand_loss:
            event("diagnostic", code="FT010", level="warn",
                  message=f"The base model looks untrained: its held-out loss ({base_val:.2f}) is about what "
                          f"random guessing gives ({rand_loss:.2f} = ln of the {len(tok)}-token vocabulary). "
                          "Training will lower the loss, but the result will not be a useful model.",
                  fix="Start from a pretrained model (e.g. an instruct checkpoint), not random weights.")
        if lead:
            model.save_pretrained(c.run_dir / "best")  # step-0 adapter == base; never ship worse than base
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    order: list[list[int]] = []
    rng = random.Random(c.seed + step + 7919 * rank)  # each rank shuffles independently
    t0, tok_count, bad, regress, spill_warned, oom_count = time.time(), 0, 0, 0, False, 0
    while step < c.max_steps:
        for g in opt.param_groups:
            g["lr"] = _lr(step, c)
        loss_acc, got = 0.0, 0
        oom = False
        try:
            while got < eps:  # micro-batches by token budget until the step has `eps` examples
                if not order:
                    order = tr.batches(budget_tokens, rng)
                idx = order.pop()[:eps - got]
                b = tr.collate(idx, device)
                w = len(idx) / eps
                with torch.autocast(device, dtype=amp, enabled=amp is not None):
                    loss = causal_lm_loss(model, b, c.ce_chunk) if c.chunked_ce else model(**b).loss
                scaler.scale(loss * w).backward()
                loss_acc += loss.item() * w
                got += len(idx)
                tok_count += int(b["attention_mask"].sum().item())
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _is_mem_error(exc):
                raise
            oom = True
        if world > 1:  # recovery must be a collective decision, or ranks fall out of step
            oom = _dist_max(1.0 if oom else 0.0, device) > 0
        if oom:
            # Adaptive recovery: the estimator cannot see cuBLAS workspaces or fragmentation, so on a
            # memory failure drop this step's partial grads, shrink the token budget and retry.
            opt.zero_grad(set_to_none=True)
            b = loss = None
            gc.collect()
            torch.cuda.empty_cache()
            oom_count += 1
            if oom_count > 6 or budget_tokens <= 512:
                event("failed", reason="out of memory")
                raise RuntimeError("Out of GPU memory even at the smallest batch. Lower --max-len "
                                   "or use --quant 4bit.")
            budget_tokens = max(512, int(budget_tokens * 0.7))
            order = []
            event("diagnostic", code="FT006", level="warn",
                  message=f"GPU memory error at step {step}; retrying with a smaller micro-batch "
                  f"(token budget now {budget_tokens:,}).",
                  fix="The planner's estimate was too optimistic for this model/GPU.")
            continue
        if world > 1:
            _allreduce_mean_grads(trainable, world)
            loss_acc = _dist_sum(loss_acc, device) / world
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, c.grad_clip).item()
        if not math.isfinite(loss_acc) or not math.isfinite(gnorm):
            bad += 1
            opt.zero_grad(set_to_none=True)
            event("diagnostic", code="FT001", level="warn", fix="",
                  message=f"Non-finite loss/grad at step {step}; skipped ({bad}/5).")
            if bad >= 5:
                event("failed", reason="training diverged")
                raise RuntimeError("Fine-tuning diverged. Lower --lr.")
            scaler.update()
            step += 1
            continue
        bad = 0
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 5 == 0 or step == 1:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
            if world > 1:
                tok_count = int(_dist_sum(tok_count, device))
            event("step", step=step, loss=loss_acc, lr=_lr(step, c), grad_norm=gnorm,
                  tok_per_s=tok_count / dt if dt > 0 else 0.0, peak_mem_gb=mem)
            t0, tok_count = time.time(), 0
            if device == "cuda" and not spill_warned and mem > 0.97 * hw.vram_gb:
                spill_warned = True
                event("diagnostic", code="FT005", level="error",
                      message=f"Peak memory {mem:.2f} GB ~ the whole {hw.vram_gb:.1f} GB GPU: the driver "
                      "may spill to system RAM and throughput can drop 10x or more.",
                      fix="Lower --max-len or --batch-size, or use --quant 4bit.")

        if step % c.eval_interval == 0 or step == c.max_steps:
            vl = eval_loss(model, va, c.eval_examples, eval_bs, device, amp, ce)
            event("eval", step=step, val_loss=vl, train_loss=loss_acc, val_ppl=math.exp(min(vl, 20)))
            if vl < best_val:
                best_val, regress = vl, 0
                if lead:
                    model.save_pretrained(c.run_dir / "best")
            else:
                regress += 1
                if regress >= 2:
                    event("diagnostic", code="FT002", level="warn",
                          message=f"Val loss worse than best for {regress} evals: overfitting.",
                          fix="Fewer steps, lower LR, or more data. The best adapter is kept.")
        if lead and (step % c.eval_interval == 0 or step == c.max_steps):
            tmp = last.with_suffix(".tmp")
            torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(),
                        "step": step, "best_val": best_val}, tmp)
            tmp.replace(last)

    div = None
    if world > 1:  # proof the replicas stayed identical: max |param - rank0's param| over all ranks
        import torch.distributed as dist

        worst = 0.0
        for p in trainable:
            ref = p.detach().clone()
            dist.broadcast(ref, 0)
            worst = max(worst, (p.detach() - ref).abs().max().item())
        div = _dist_max(worst, device)
    summary = {"steps": step, "best_val_loss": best_val, "best_val_ppl": math.exp(min(best_val, 20)),
               "trainable_params": n_train,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0}
    summary["world_size"] = world
    if div is not None:
        summary["ddp_max_param_divergence"] = div
    event("finished", **summary)
    log.close()
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return summary
