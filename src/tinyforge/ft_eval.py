"""Evaluation for fine-tuned adapters, always *relative to the base model*.

Gates: held-out improvement, catastrophic-forgetting, output sanity, merge equivalence, speed/VRAM.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from . import hardware
from .finetune import ChatDataset, FTConfig, eval_loss, load_base, load_tokenizer
from .warnings import Level, Report

# Neutral general-domain text for the forgetting check (not in any fine-tuning set).
GENERAL_TEXT = [
    "The water cycle describes how water evaporates from the surface of the earth, rises into the "
    "atmosphere, cools and condenses into clouds, and falls again as precipitation. Some of this water "
    "flows over land as runoff, while the rest soaks into the ground and replenishes aquifers.",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. Neil Armstrong and Buzz Aldrin "
    "spent about two and a half hours outside the lunar module while Michael Collins orbited above in "
    "the command module.",
    "A binary search algorithm finds a target value in a sorted array by repeatedly dividing the search "
    "interval in half. If the middle element is smaller than the target, the search continues in the "
    "upper half; otherwise it continues in the lower half.",
    "Photosynthesis is the process by which green plants use sunlight to synthesise nutrients from carbon "
    "dioxide and water. It takes place mainly in the leaves, where chlorophyll absorbs light energy.",
    "The French Revolution began in 1789 and led to the end of the monarchy in France. It reshaped "
    "politics across Europe by promoting ideas about citizenship, equality and the rights of man.",
]
PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "Give me three tips for staying focused while studying.",
    "Write a short, polite email declining a meeting invitation.",
    "What is the difference between a list and a tuple in Python?",
]


def _distinct2(text: str) -> float:
    w = text.split()
    g = list(zip(w, w[1:], strict=False))
    return len(set(g)) / len(g) if g else 0.0


@torch.no_grad()
def _general_loss(model, tok, device: str, amp) -> float:
    model.eval()
    tot, n = 0.0, 0
    for t in GENERAL_TEXT:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        with torch.autocast(device, dtype=amp, enabled=amp is not None):
            out = model(input_ids=ids, labels=ids)
        tot += out.loss.item() * (ids.size(1) - 1)
        n += ids.size(1) - 1
    return tot / n


@torch.no_grad()
def _gen(model, tok, prompt: str, device: str, max_new: int = 96) -> str:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.input_ids.size(1):], skip_special_tokens=True).strip()


def evaluate(run_dir: Path, data_dir: Path | None = None, merge: bool = True) -> dict:
    from peft import PeftModel

    cfg = FTConfig.model_validate_json((run_dir / "ft_config.json").read_text())
    data_dir = data_dir or cfg.data_dir
    hw = hardware.probe()
    device = hw.device
    amp = (torch.bfloat16 if hw.bf16_supported else torch.float16) if device == "cuda" else None
    tok = load_tokenizer(cfg.base_model)
    val = ChatDataset(data_dir / "val.jsonl", tok, cfg.max_len)
    rep = Report()
    res: dict = {"run": str(run_dir), "base_model": cfg.base_model, "quant": cfg.quant,
                 "val_examples": len(val)}

    base = load_base(cfg.base_model, cfg.quant, hw.bf16_supported, device)
    model = PeftModel.from_pretrained(base, run_dir / "best")
    n = max(len(val), 1)
    bs = max(1, min(cfg.batch_size, 4))

    # 1. Held-out loss: tuned vs base (adapter switched off)
    tuned = eval_loss(model, val, n, bs, device, amp)
    with model.disable_adapter():
        base_loss = eval_loss(model, val, n, bs, device, amp)
    res.update(val_loss_base=base_loss, val_loss_tuned=tuned,
               val_ppl_base=math.exp(min(base_loss, 20)), val_ppl_tuned=math.exp(min(tuned, 20)),
               improvement_pct=100 * (base_loss - tuned) / base_loss)
    if tuned >= base_loss:
        rep.add("FE001", Level.ERROR, f"Tuned model is not better than base on held-out data "
                f"({tuned:.3f} vs {base_loss:.3f}).", "Check data quality, raise steps, or lower LR.")
    elif res["improvement_pct"] < 2:
        rep.add("FE002", Level.WARN, f"Only {res['improvement_pct']:.1f}% improvement over base: "
                "the data may not teach anything new.")

    # 2. Catastrophic forgetting on general text
    g_tuned = _general_loss(model, tok, device, amp)
    with model.disable_adapter():
        g_base = _general_loss(model, tok, device, amp)
    drift = 100 * (g_tuned - g_base) / g_base
    res.update(general_loss_base=g_base, general_loss_tuned=g_tuned, forgetting_pct=drift)
    if drift > 25:
        rep.add("FE003", Level.ERROR, f"General-text loss rose {drift:.0f}%: severe forgetting.",
                "Lower LR/rank/steps.")
    elif drift > 10:
        rep.add("FE004", Level.WARN, f"General-text loss rose {drift:.0f}% (forgetting).",
                "Consider a lower LR or fewer steps.")

    # 3. Generation sanity, base vs tuned, greedy so results are comparable and reproducible
    samples, empties, d2s, lens = [], 0, [], []
    for p in PROMPTS:
        t_out = _gen(model, tok, p, device)
        with model.disable_adapter():
            b_out = _gen(model, tok, p, device)
        samples.append({"prompt": p, "base": b_out, "tuned": t_out})
        empties += not t_out.strip()
        d2s.append(_distinct2(t_out))
        lens.append(len(t_out.split()))
    res.update(samples=samples, empty_rate=empties / len(PROMPTS), distinct_2=sum(d2s) / len(d2s),
               mean_words=sum(lens) / len(lens))
    if empties:
        rep.add("FE005", Level.ERROR, f"{empties}/{len(PROMPTS)} generations were empty.")
    if res["distinct_2"] < 0.6:
        rep.add("FE006", Level.WARN, f"Repetitive outputs (distinct-2 {res['distinct_2']:.2f}).")
    if all(s["base"] == s["tuned"] for s in samples):
        rep.add("FE007", Level.WARN, "Tuned and base outputs are identical: the adapter has no visible "
                "effect.")

    # 4. Speed / memory
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    text = _gen(model, tok, PROMPTS[0], device, max_new=64)
    if device == "cuda":
        torch.cuda.synchronize()
    ntok = len(tok(text).input_ids)
    res.update(decode_tok_per_s=ntok / (time.perf_counter() - t),
               inference_peak_mem_gb=torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0)

    # 5. Merge adapter into base weights -> standalone deployable model, verified equivalent
    if merge and cfg.quant == "none":
        ids = tok(PROMPTS[1], return_tensors="pt").input_ids.to(device)
        model.eval()
        with torch.no_grad():
            before = model(input_ids=ids).logits[:, -1].float()
            merged = model.merge_and_unload()
            after = merged(input_ids=ids).logits[:, -1].float()
        err = (before - after).abs().max().item()
        res["merge_max_logit_err"] = err
        out = run_dir / "merged"
        merged.save_pretrained(out)
        tok.save_pretrained(out)
        res["merged_dir"] = str(out)
        if err > 0.5:  # bf16 round-off scale; larger means the merge is wrong
            rep.add("FE008", Level.ERROR, f"Merged model deviates from adapter model (max logit err "
                    f"{err:.2f}).")
    elif merge:
        rep.add("FE009", Level.INFO, "Merge skipped: 4-bit base. Serve base+adapter, or retrain with "
                "--quant none to export a merged model.")

    res["diagnostics"] = rep.to_list()
    res["passed"] = not rep.has_errors
    (run_dir / "eval.json").write_text(json.dumps(res, indent=2))
    return res
