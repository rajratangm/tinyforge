"""Greedy generation with kernels off vs auto: how far do the outputs drift?

    .venv\\Scripts\\python tools/bench_generation.py runs/kernel-bench/auto benchmarks/kernels-generation.json

bf16 maths is not associative and the kernels round in slightly different places, so token-for-token equality
is not guaranteed. This records the real numbers: the largest logit difference on the prompt, and for each
prompt the position where the greedy continuations first differ (None = identical).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "Write a SQL query that counts the rows in a table called orders.",
    "List three differences between a process and a thread.",
    "What is the capital of France, and what river runs through it?",
    "Give me a short checklist for reviewing a pull request.",
    "Translate 'good morning, how are you?' into Spanish.",
    "Why is the sky blue?",
    "Summarise the plot of Romeo and Juliet in one paragraph.",
]


def main(run_dir: Path, out: Path, max_new: int = 80) -> None:
    from peft import PeftModel

    from tinyforge import hardware
    from tinyforge.finetune import FTConfig, load_base, load_tokenizer
    from tinyforge.kernels.patch import apply_kernels

    cfg = FTConfig.model_validate_json((run_dir / "ft_config.json").read_text())
    hw = hardware.probe()
    model = PeftModel.from_pretrained(load_base(cfg.base_model, cfg.quant, hw.bf16_supported, hw.device),
                                      run_dir / "best").eval()
    tok = load_tokenizer(cfg.base_model)

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        return tok(text, return_tensors="pt", add_special_tokens=False).to(hw.device)

    @torch.no_grad()
    def run(p):
        ids = encode(p)
        logits = model(**ids).logits[:, -1].float()
        gen = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        return logits, gen[0, ids.input_ids.size(1):].tolist()

    off = [run(p) for p in PROMPTS]
    report = apply_kernels(model, "auto", inference=True)
    on = [run(p) for p in PROMPTS]
    rows = []
    for p, (lo, go), (ln, gn) in zip(PROMPTS, off, on, strict=True):
        first = next((i for i, (a, b) in enumerate(zip(go, gn, strict=False)) if a != b), None)
        if first is None and len(go) != len(gn):
            first = min(len(go), len(gn))
        rows.append({"prompt": p, "max_logit_diff": float((lo - ln).abs().max()), "first_divergence": first,
                     "tokens": len(go)})
    result = {"model": cfg.base_model, "max_new_tokens": max_new, "kernels": report.as_dict(),
              "identical_prompts": sum(r["first_divergence"] is None for r in rows), "prompts": len(rows),
              "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    for r in rows:
        print(r["first_divergence"], round(r["max_logit_diff"], 4), r["prompt"][:50])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
