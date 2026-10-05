"""Typed configuration, presets and a VRAM-aware planner."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .warnings import Level, Report


class ModelConfig(BaseModel):
    vocab_size: int = 2048
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    mlp_mult: float = 8 / 3  # SwiGLU hidden = mult * d_model (rounded to 64)
    dropout: float = 0.0
    tie_embeddings: bool = True

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2:
            raise ValueError("head dim must be even for RoPE")
        return self

    @property
    def hidden(self) -> int:
        return int(round(self.d_model * self.mlp_mult / 64) * 64)

    def num_params(self) -> int:
        d, h, v = self.d_model, self.hidden, self.vocab_size
        per_layer = 4 * d * d + 3 * d * h + 2 * d
        emb = v * d
        head = 0 if self.tie_embeddings else v * d
        return self.n_layer * per_layer + emb + head + d


class TrainConfig(BaseModel):
    run_dir: Path = Path("runs/default")
    data_dir: Path = Path("data/tinyshakespeare")
    max_steps: int = 2000
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    ckpt_interval: int = 500
    precision: str = Field("auto", pattern="^(auto|bf16|fp16|fp32)$")
    grad_checkpointing: bool = False
    compile: bool = False
    seed: int = 1337

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum


# name -> (n_layer, n_head, d_model)
PRESETS: dict[str, tuple[int, int, int]] = {
    "nano": (4, 4, 128),    # ~1M params, CPU-friendly smoke tests
    "micro": (6, 6, 384),   # ~12M
    "small": (8, 8, 512),   # ~28M
    "base": (12, 12, 768),  # ~100M, needs 12+ GB
}


def make_model_config(preset: str, vocab_size: int, block_size: int) -> ModelConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    n_layer, n_head, d_model = PRESETS[preset]
    return ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=n_layer, n_head=n_head, d_model=d_model)


def estimate_train_vram_gb(m: ModelConfig, micro_batch: int, checkpointing: bool,
                           half_precision: bool = True) -> float:
    """Rough upper-bound estimate: fp32 master weights + grads + AdamW + activations + logits."""
    p = m.num_params()
    states = p * (4 + 4 + 8)
    act_bytes = 2 if half_precision else 4
    tokens = micro_batch * m.block_size
    if checkpointing:
        acts = tokens * m.d_model * m.n_layer * act_bytes + tokens * m.d_model * 34 * act_bytes
    else:
        acts = tokens * m.d_model * m.n_layer * 34 * act_bytes / 2
    logits = tokens * m.vocab_size * 4 * 2
    overhead = 0.6 * 1024**3  # CUDA context + allocator fragmentation
    return (states + acts + logits + overhead) / 1024**3


def plan(model: ModelConfig, train: TrainConfig, vram_gb: float,
         bf16: bool) -> tuple[TrainConfig, Report]:
    """Adjust batch/accum/checkpointing so training fits in VRAM, and explain every change."""
    r = Report()
    t = train.model_copy()
    if t.precision == "auto":
        t.precision = "bf16" if bf16 else "fp16" if vram_gb > 0 else "fp32"
        r.add("PL001", Level.INFO, f"Precision auto-selected: {t.precision}.")
    if vram_gb <= 0:
        return t, r  # CPU: nothing to fit
    budget = vram_gb * 0.85
    target_tokens = t.batch_size * t.grad_accum
    half = t.precision != "fp32"
    while True:
        need = estimate_train_vram_gb(model, t.batch_size, t.grad_checkpointing, half)
        if need <= budget:
            break
        if not t.grad_checkpointing:
            t.grad_checkpointing = True
            r.add("PL002", Level.INFO,
                  f"Estimated {need:.1f} GB > budget {budget:.1f} GB: enabled gradient "
                  "checkpointing (~30% slower, much less memory).")
        elif t.batch_size > 1:
            t.batch_size //= 2
            t.grad_accum = max(1, target_tokens // t.batch_size)
            r.add("PL003", Level.WARN,
                  f"Reduced micro-batch to {t.batch_size} (grad_accum={t.grad_accum}) "
                  "to fit VRAM; effective batch size is preserved.")
        else:
            r.add("PL004", Level.ERROR,
                  f"Model needs ~{need:.1f} GB even at batch size 1; only {vram_gb:.1f} GB "
                  "available.", "Choose a smaller preset or shorter --block-size.")
            break
    need = estimate_train_vram_gb(model, t.batch_size, t.grad_checkpointing, half)
    r.add("PL005", Level.INFO,
          f"Plan: {model.num_params()/1e6:.1f}M params, micro-batch {t.batch_size} x "
          f"accum {t.grad_accum}, est. {need:.1f} GB of {vram_gb:.1f} GB.")
    return t, r
