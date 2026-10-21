"""Registry of kernels: each pairs a PyTorch reference with a Triton implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class Target:
    """Where a (user) kernel plugs into a model: which modules it replaces and with what forward."""

    match: Callable[[torch.nn.Module], bool]
    # shape to verify and tune on, taken from the first matching module
    shape_for: Callable[[torch.nn.Module], tuple[int, ...]]
    # (module, tuned params) -> forward(*args, **kwargs); return NotImplemented to use the original forward
    make_forward: Callable[[torch.nn.Module, dict], Callable]


@dataclass
class KernelSpec:
    name: str
    reference: Callable  # pure PyTorch op, the ground truth
    kernel: Callable  # same positional args as reference, plus **tune from `tunables`
    dtypes: tuple[torch.dtype, ...]
    supports: Callable[[tuple[int, ...]], bool]
    tunables: dict[str, list[int]] = field(default_factory=dict)
    make_inputs: Callable[[tuple[int, ...], torch.dtype, str], tuple[torch.Tensor, ...]] | None = None
    memory_kernel: bool = False  # adopted for its memory saving even when not faster
    needs_grad: bool = True  # False: inference-only kernel, verified and timed forward-only
    example_shape: tuple[int, ...] = ()  # what `kernels verify` runs on when no model supplies a shape
    target: Target | None = None  # set by user-defined kernels so the patcher knows where to apply them


_SPECS: dict[str, KernelSpec] = {}


def register(spec: KernelSpec) -> None:
    if spec.name in _SPECS:
        raise ValueError(f"kernel {spec.name!r} is already registered")
    _SPECS[spec.name] = spec


def get(name: str) -> KernelSpec:
    return _SPECS[name]


def all_specs() -> list[KernelSpec]:
    return list(_SPECS.values())
