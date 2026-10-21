"""Registry of kernels: each pairs a PyTorch reference with a Triton implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


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


_SPECS: dict[str, KernelSpec] = {}


def register(spec: KernelSpec) -> None:
    if spec.name in _SPECS:
        raise ValueError(f"kernel {spec.name!r} is already registered")
    _SPECS[spec.name] = spec


def get(name: str) -> KernelSpec:
    return _SPECS[name]


def all_specs() -> list[KernelSpec]:
    return list(_SPECS.values())
