"""Inference: KV-cache generation, sampling, int8 weight-only quantization, Triton toggle."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GPT, RMSNorm


class Int8Linear(nn.Module):
    """Per-output-channel symmetric int8 weights, dequantised on the fly (~4x less weight memory)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.squeeze(1).to(lin.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        return F.linear(x, w)


def quantize_int8(model: GPT) -> GPT:
    """Quantise all transformer-block linears in place. Embedding/head stay full precision."""
    for blk in model.blocks:
        for parent in (blk.attn, blk.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Int8Linear(child))
    return model


def enable_triton(on: bool = True) -> bool:
    if not on:
        RMSNorm.use_triton = False
        return False
    try:
        import triton  # noqa: F401

        RMSNorm.use_triton = True
    except ImportError:
        RMSNorm.use_triton = False
    return RMSNorm.use_triton


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        cum = F.softmax(sl, dim=-1).cumsum(-1)
        drop = cum - F.softmax(sl, dim=-1) > top_p
        sl = sl.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


@torch.no_grad()
def generate_ids(model: GPT, prompt: list[int], max_new: int = 200, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.95, eos_id: int | None = None,
                 seed: int | None = None) -> Iterator[int]:
    """Yield token ids one at a time using a KV cache (O(T) per token instead of O(T^2))."""
    model.eval()
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    prompt = prompt[-(model.c.block_size - 1):] or [0]
    caches = [{} for _ in model.blocks]
    idx = torch.tensor([prompt], device=device)
    pos = 0
    for _ in range(max_new):
        if pos + idx.size(1) >= model.c.block_size:
            return  # context window full
        logits, _ = model(idx, caches=caches, start_pos=pos)
        pos += idx.size(1)
        nxt = _sample(logits[:, -1, :].float(), temperature, top_k, top_p)
        tok = int(nxt.item())
        if eos_id is not None and tok == eos_id:
            return
        yield tok
        idx = nxt


def generate_text(model: GPT, tokenizer, prompt: str, **kw) -> str:
    ids = tokenizer.encode(prompt).ids if prompt else [tokenizer.token_to_id("<bos>") or 0]
    out = list(generate_ids(model, ids, eos_id=tokenizer.token_to_id("<eos>"), **kw))
    return tokenizer.decode(out)
