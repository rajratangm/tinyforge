"""Small modern decoder-only transformer: RMSNorm, RoPE, SwiGLU, SDPA/flash attention, KV cache."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Uses the fused Triton kernel for inference on CUDA when enabled, else PyTorch."""

    use_triton = False  # flipped by infer.enable_triton()

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if RMSNorm.use_triton and x.is_cuda and not torch.is_grad_enabled():
            from .kernels.rmsnorm import rmsnorm_triton

            return rmsnorm_triton(x, self.weight, self.eps)
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def rope_tables(head_dim: int, max_len: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D/2)
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.h = c.n_head
        self.hd = c.d_model // c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.dropout = c.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # Causal mask only when query and key lengths match (prefill / training).
        causal = cache is None or q.size(2) == k.size(2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.hidden, bias=False)
        self.up = nn.Linear(c.d_model, c.hidden, bias=False)
        self.down = nn.Linear(c.hidden, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.attn, self.mlp = Attention(c), SwiGLU(c)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.head.weight = self.tok.weight
        self.grad_checkpointing = False
        self._rope: tuple | None = None
        self.apply(self._init)
        for n, p in self.named_parameters():  # GPT-2 style residual scaling
            if n.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def _rope_for(self, device):
        if self._rope is None or self._rope[0].device != device:
            self._rope = rope_tables(self.c.d_model // self.c.n_head, self.c.block_size, device)
        return self._rope

    def forward(self, idx, targets=None, caches=None, start_pos: int = 0):
        B, T = idx.shape
        if start_pos + T > self.c.block_size:
            raise ValueError(f"sequence {start_pos + T} exceeds block_size {self.c.block_size}")
        cos, sin = self._rope_for(idx.device)
        cos, sin = cos[start_pos:start_pos + T], sin[start_pos:start_pos + T]
        x = self.tok(idx)
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and self.training and caches is None:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin, None if caches is None else caches[i])
        x = self.norm(x)
        if targets is None:
            return self.head(x[:, -1:, :]), None  # inference only needs the last position
        logits = self.head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
