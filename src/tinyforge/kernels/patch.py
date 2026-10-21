"""Swap verified, tuned Triton kernels into a loaded HF causal LM; any failure restores the original."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import registry
from .env import triton_status
from .tune import tune

KERNELS = ("rmsnorm", "swiglu", "rope", "add_rmsnorm")
LAYER_ATTRS = ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp")
LAYER_CLASSES = {"Qwen2DecoderLayer", "LlamaDecoderLayer", "MistralDecoderLayer"}
ROPE_FN = "apply_rotary_pos_emb"
_FALLBACK = object()  # a patched forward returns this for inputs it does not handle: use the original


@dataclass
class KernelReport:
    patched: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    speedups: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class _Patches:
    """Undo actions per kernel, so one failing kernel is removed everywhere and the others keep running."""

    def __init__(self, report: KernelReport):
        self.report = report
        self.undo: dict[str, list[Callable[[], None]]] = {}
        self.enabled = True  # False: every patched forward passes straight through to the original

    def add(self, kernel: str, undo: Callable[[], None]) -> None:
        self.undo.setdefault(kernel, []).append(undo)
        if kernel not in self.report.patched:
            self.report.patched.append(kernel)

    def restore(self, kernel: str, why: str | None = None) -> None:
        for undo in self.undo.pop(kernel, []):
            undo()
        if kernel in self.report.patched:
            self.report.patched.remove(kernel)
        if why:
            self.report.skipped[kernel] = why

    def restore_all(self, why: str | None = None) -> None:
        for kernel in list(self.undo):
            self.restore(kernel, why)


def _is_torch_control_flow(e: BaseException) -> bool:
    """Non-reentrant activation checkpointing stops its recompute early by raising an exception through the
    forward being recomputed. That is not a kernel failure and must reach torch untouched."""
    return type(e).__module__.startswith("torch.utils.checkpoint")


def _wrap_forward(module: torch.nn.Module, kernel: str, new: Callable, patches: _Patches) -> None:
    orig = module.forward
    had_instance_forward = "forward" in module.__dict__

    def wrapped(*args, **kwargs):
        if not patches.enabled:
            return orig(*args, **kwargs)
        try:
            out = new(*args, **kwargs)
        except Exception as e:
            if _is_torch_control_flow(e):
                raise
            patches.restore(kernel, f"{type(e).__name__}: {e}"[:300])
            return orig(*args, **kwargs)
        return orig(*args, **kwargs) if out is _FALLBACK else out

    def undo():
        if had_instance_forward:
            module.forward = orig
        else:
            module.__dict__.pop("forward", None)

    module.forward = wrapped
    patches.add(kernel, undo)


def _is_rmsnorm(m: torch.nn.Module) -> bool:
    return type(m).__name__.endswith("RMSNorm") and hasattr(m, "weight") and hasattr(m, "variance_epsilon")


def _is_swiglu_mlp(m: torch.nn.Module) -> bool:
    if not all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj", "act_fn")):
        return False
    act = m.act_fn
    return (isinstance(act, torch.nn.SiLU) or type(act).__name__ in {"SiLU", "SiLUActivation"}
            or act is torch.nn.functional.silu)


def _is_layer(m: torch.nn.Module) -> bool:
    return (type(m).__name__ in LAYER_CLASSES
            and all(hasattr(m, a) for a in LAYER_ATTRS))


def _decide(spec, shape: tuple[int, ...], dtype: torch.dtype, mode: str, report: KernelReport,
            key: str | None = None):
    """Verify + tune `spec` (a KernelSpec or a registered name); returns its params if it should be used."""
    spec = registry.get(spec) if isinstance(spec, str) else spec
    key = key or spec.name
    try:
        r = tune(spec, shape, dtype, "cuda")
    except Exception as e:
        report.skipped[key] = f"tuning failed: {type(e).__name__}: {e}"[:300]
        return None
    if r.adopted or (mode == "force" and r.kernel_ms > 0):
        report.speedups[key] = round(r.ref_ms / r.kernel_ms, 3) if r.kernel_ms else 0.0
        return r.params
    report.skipped[key] = r.reason or "not adopted"
    return None


def _probe_logits(base: torch.nn.Module) -> torch.Tensor:
    device = next(base.parameters()).device
    g = torch.Generator().manual_seed(7)
    x = torch.randint(0, base.config.vocab_size, (1, 16), generator=g).to(device)
    was_training = base.training
    base.eval()
    try:
        with torch.no_grad():
            return base(input_ids=x).logits.float()
    finally:
        base.train(was_training)


def apply_kernels(model: torch.nn.Module, mode: str = "auto", inference: bool = False,
                  user_paths: list[Path] | None = None) -> KernelReport:
    """mode: off | auto (adopt only what is faster or saves memory) | force (adopt anything that verifies).
    user_paths: files/folders of user-defined kernels to load (explicit only, see kernels/user.py)."""
    report = KernelReport()
    model._tinyforge_kernel_report = report
    if mode == "off":
        return report
    status = triton_status()
    if not status.ok:
        report.skipped.update(dict.fromkeys(KERNELS, status.reason))
        return report
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    first = next(base.parameters(), None)
    if first is None or not first.is_cuda:
        report.skipped.update(dict.fromkeys(KERNELS, "model is not on a CUDA device"))
        return report

    from . import add_rmsnorm, ce, rmsnorm, rope, swiglu  # noqa: F401  (import registers the specs)

    patches = _Patches(report)
    model._tinyforge_patches = patches
    cfg = base.config
    ref_logits = _probe_logits(base) if inference else None

    norms = [m for m in base.modules() if _is_rmsnorm(m)]
    dtype = norms[0].weight.dtype if norms else (getattr(cfg, "dtype", None) or torch.float32)
    if not isinstance(dtype, torch.dtype):
        dtype = torch.float32

    def norm_forward(m, params):
        def new(x):
            w = m.weight
            if not (x.is_cuda and x.dtype == w.dtype and w.dim() == 1 and x.shape[-1] == w.shape[0]):
                return _FALLBACK
            return rmsnorm.rmsnorm_fn(x, w, m.variance_epsilon, params["num_warps"])
        return new

    by_width: dict[int, dict | None] = {}
    for m in norms:
        w = m.weight.shape[0]
        if w not in by_width:
            by_width[w] = _decide("rmsnorm", (2048, w), m.weight.dtype, mode, report)
        if by_width[w] is not None and m.weight.dim() == 1:
            _wrap_forward(m, "rmsnorm", norm_forward(m, by_width[w]), patches)

    mlps = [m for m in base.modules() if _is_swiglu_mlp(m)]
    params = _decide("swiglu", (2048, cfg.intermediate_size), dtype, mode, report) if mlps else None
    if params is not None:
        def mlp_forward(m, params):
            def new(x):
                gate, up = m.gate_proj(x), m.up_proj(x)
                if not (gate.is_cuda and gate.dtype == up.dtype and gate.shape == up.shape):
                    return _FALLBACK
                return m.down_proj(swiglu.swiglu_fn(gate, up, params["BLOCK"], params["num_warps"]))
            return new
        for m in mlps:
            _wrap_forward(m, "swiglu", mlp_forward(m, params), patches)

    heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    rope_modules = {sys.modules[type(m).__module__] for m in base.modules()
                    if type(m).__name__.endswith("Attention")
                    and hasattr(sys.modules.get(type(m).__module__), ROPE_FN)}
    params = None
    if rope_modules:
        params = _decide("rope", (1, heads, kv_heads, 256, head_dim), dtype, mode, report)
    if params is not None:
        for mod in rope_modules:
            _patch_rope_global(mod, params, rope, patches)

    layers = [m for m in base.modules() if _is_layer(m)] if inference else []
    params = _decide("add_rmsnorm", (256, cfg.hidden_size), dtype, mode, report) if layers else None
    if params is not None:
        for m in layers:
            _wrap_forward(m, "add_rmsnorm", _layer_forward(m, params, add_rmsnorm), patches)

    if user_paths:
        _apply_user_kernels(base, user_paths, dtype, mode, report, patches)

    if inference and patches.undo:
        tol = (2e-3 if dtype == torch.float32 else 6e-2) * max(1.0, ref_logits.abs().max().item())
        try:
            ok = (ref_logits - _probe_logits(base)).abs().max().item() <= tol
        except Exception:
            ok = False
        if not ok:
            patches.restore_all("end-to-end logits check failed after patching")
    return report


def _apply_user_kernels(base, user_paths, dtype, mode, report: KernelReport, patches: _Patches) -> None:
    from .user import load_user_kernels

    specs, errors = load_user_kernels(list(user_paths))
    for path, err in errors.items():
        report.skipped[f"user:{Path(path).name}"] = f"could not load: {err}"
    for spec in specs:
        key = f"user:{spec.name}"
        if spec.target is None:
            report.skipped[key] = "the kernel defines no target, so there is nothing to patch (verify-only)"
            continue
        try:
            matches = [m for m in base.modules() if spec.target.match(m)]
            if not matches:
                report.skipped[key] = "no module in this model matches the kernel's target"
                continue
            params = _decide(spec, spec.target.shape_for(matches[0]), dtype, mode, report, key)
            if params is None:
                continue
            forwards = [spec.target.make_forward(m, params) for m in matches]
        except Exception as e:  # match / shape_for / make_forward are the user's code: report, never crash
            report.skipped[key] = f"{type(e).__name__}: {e}"[:300]
            continue
        for m, fwd in zip(matches, forwards, strict=True):
            _wrap_forward(m, key, _user_forward(fwd), patches)


def _user_forward(fwd):
    def new(*args, **kwargs):
        out = fwd(*args, **kwargs)
        return _FALLBACK if out is NotImplemented else out
    return new


def _patch_rope_global(mod, params: dict, rope_mod, patches: _Patches) -> None:
    orig = mod.apply_rotary_pos_emb

    def new(q, k, cos, sin, unsqueeze_dim=1):
        if (not patches.enabled or unsqueeze_dim != 1 or q.dim() != 4 or not q.is_cuda
                or not (q.dtype == k.dtype == cos.dtype == sin.dtype)):
            return orig(q, k, cos, sin, unsqueeze_dim)
        try:
            return rope_mod.rope_fn(q, k, cos, sin, 1, params["num_warps"])
        except Exception as e:
            if _is_torch_control_flow(e):
                raise
            patches.restore("rope", f"{type(e).__name__}: {e}"[:300])
            return orig(q, k, cos, sin, unsqueeze_dim)

    mod.apply_rotary_pos_emb = new
    patches.add("rope", lambda: setattr(mod, "apply_rotary_pos_emb", orig))


def _layer_forward(layer, params: dict, add_mod):
    """Decoder layer for decoding: the residual add after attention is fused into the next RMSNorm."""
    def new(hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
            position_embeddings=None, **kwargs):
        norm = layer.post_attention_layernorm
        if (torch.is_grad_enabled() or not hidden_states.is_cuda or hidden_states.dtype != norm.weight.dtype):
            return _FALLBACK
        residual = hidden_states
        h = layer.input_layernorm(hidden_states)
        h, _ = layer.self_attn(hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=past_key_values, use_cache=use_cache,
                               position_embeddings=position_embeddings, **kwargs)
        normed, residual = add_mod.add_rmsnorm(h, residual, norm.weight, norm.variance_epsilon,
                                               params["num_warps"])
        return residual + layer.mlp(normed)
    return new


def restore_kernels(model: torch.nn.Module) -> None:
    """Put every original forward back (used by the drift guard and by tests)."""
    patches = getattr(model, "_tinyforge_patches", None)
    if patches is not None:
        patches.restore_all()


def describe_matches(model: torch.nn.Module) -> dict[str, int]:
    """How many modules each kernel could replace in this model (works on meta-device models)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    mods = list(base.modules())
    rope_mods = {type(m).__module__ for m in mods if type(m).__name__.endswith("Attention")
                 and hasattr(sys.modules.get(type(m).__module__), "apply_rotary_pos_emb")}
    return {"rmsnorm": sum(_is_rmsnorm(m) for m in mods), "swiglu": sum(_is_swiglu_mlp(m) for m in mods),
            "rope": len(rope_mods), "add_rmsnorm": sum(_is_layer(m) for m in mods)}


def kernels_agree(model: torch.nn.Module, loss_fn: Callable[[bool], torch.Tensor], tol: float = 0.02) -> bool:
    """Drift guard: the loss with kernels must match the loss without them (relative `tol`).
    loss_fn(use_kernels) computes one loss; the patched forwards are bypassed while it runs with False."""
    patches = getattr(model, "_tinyforge_patches", None)
    if patches is None or not patches.undo:
        return True
    with torch.no_grad():
        with_kernels = float(loss_fn(True))
        patches.enabled = False
        try:
            reference = float(loss_fn(False))
        finally:
            patches.enabled = True
    return abs(with_kernels - reference) <= tol * max(1.0, abs(reference))


def decide_fused_ce(model: torch.nn.Module, mode: str, report: KernelReport) -> dict | None:
    """Tune the fused cross-entropy for this model's LM head; None keeps the PyTorch chunked path.
    It switches the loss function rather than patching a module, so there is nothing to undo."""
    if mode == "off":
        return None
    if not triton_status().ok:
        report.skipped["fused_ce"] = "Triton is not usable here"
        return None
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.get_output_embeddings()
    if head is None or not head.weight.is_floating_point() or not head.weight.is_cuda:
        report.skipped["fused_ce"] = "LM head is quantised or not on a CUDA device"
        return None
    from . import ce  # noqa: F401  (import registers the spec)

    # a proxy width keeps tuning inside a 4 GB card; vocab is the dimension the kernel loops over
    params = _decide("fused_ce", (256, min(base.config.hidden_size, 512), base.config.vocab_size),
                     head.weight.dtype, mode, report)
    if params is not None:
        patches = getattr(model, "_tinyforge_patches", None)
        if patches is not None:
            patches.add("fused_ce", lambda: None)
        elif "fused_ce" not in report.patched:
            report.patched.append("fused_ce")
    return params
