"""Memory-ladder building blocks: techniques that cut training memory without changing the maths.

Rung 1 - chunked cross-entropy. A causal LM normally materialises logits of shape [batch, seq, vocab] in fp32
(for a 152k-vocab model at 4k tokens that is ~2.5 GB, plus its gradient). Only positions inside the assistant
reply carry a label, so we (a) run the transformer body alone, (b) gather just the labelled hidden states, and
(c) project them through the LM head in chunks under activation checkpointing so no chunk's logits survive the
forward pass. The loss equals the mean token cross-entropy HF computes (see tests/test_memory.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce_sum(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                  targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight, bias).float()
    return F.cross_entropy(logits, targets, reduction="sum")


def chunked_ce_loss(hidden: torch.Tensor, lm_head: torch.nn.Module, labels: torch.Tensor,
                    chunk: int = 2048, ignore_index: int = -100) -> torch.Tensor:
    """Mean next-token cross-entropy. hidden: [B,T,H]; labels: [B,T] (unshifted, HF convention)."""
    h = hidden[:, :-1, :]
    y = labels[:, 1:]
    keep = y != ignore_index
    n = int(keep.sum().item())
    if n == 0:
        return hidden.sum() * 0.0  # keeps the graph alive; zero gradient, mirrors an all-masked batch
    h, y = h[keep], y[keep]  # [N,H], [N]
    total = hidden.new_zeros((), dtype=torch.float32)
    weight, bias = lm_head.weight, getattr(lm_head, "bias", None)
    for i in range(0, n, chunk):
        hc, yc = h[i:i + chunk], y[i:i + chunk]
        if torch.is_grad_enabled() and hc.requires_grad:
            total = total + checkpoint(_chunk_ce_sum, hc, weight, bias, yc, use_reentrant=False)
        else:
            total = total + _chunk_ce_sum(hc, weight, bias, yc)
    return total / n


def causal_lm_loss(model, batch: dict, chunk: int = 2048) -> torch.Tensor:
    """Loss for a (PEFT-wrapped) HF causal LM via the chunked path. Works for any model exposing
    get_decoder() and get_output_embeddings() (the HF contract: Llama, Qwen, Mistral, Gemma, Phi ...)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = base.get_decoder()(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    return chunked_ce_loss(out.last_hidden_state, base.get_output_embeddings(), batch["labels"], chunk)
