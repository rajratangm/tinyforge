"""Generate Triton kernels (forward and backward) for chains of elementwise ops and row reductions.

A chain is a list of ChainNode in topological order. A node input is an int (the output of an earlier node) or
"in0", "in1", ... (an external tensor; all externals share the shape of the result). Row reductions (mean, sum
over the last dimension) produce a scalar per row that later nodes broadcast over the row.

The generated source is written to <cache>/generated/<hash>.py (the hash covers the source text itself, so a
changed generator never reuses a stale kernel) and is meant to be read. Every generated kernel still goes
through the verifier, autotuner and patcher like a hand-written one; the verifier is what catches a bad
derivative or a bad code path here.

Not supported: broadcasting externals (e.g. a weight vector shared across rows), matmul, non-last-dim
reductions, data-dependent control flow.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .registry import KernelSpec

GENERATOR_VERSION = 1
_C = "0.7978845608028654"  # sqrt(2/pi)


class UnsupportedOpError(ValueError):
    """The chain uses an op the generator cannot express in Triton."""


@dataclass
class ChainNode:
    op: str
    inputs: list = field(default_factory=list)  # ints (earlier node) or "in<k>" (external tensor)
    attrs: dict = field(default_factory=dict)  # const: {"value": float}


@dataclass
class OpDef:
    arity: int
    fwd: Callable[[list[str]], str] | None  # Triton expression from operand expressions
    bwd: Callable[[str, list[str], str], list[str]] | None  # (grad, operands, output) -> grad per operand
    torch_fn: Callable | None
    positive: bool = False  # needs inputs > 0 (log, sqrt, rsqrt, div), used to build test inputs
    kind: str = "elementwise"  # elementwise | const | reduce


def _tanh(a: str) -> str:
    return f"(2.0 * tl.sigmoid(2.0 * {_C} * ({a} + 0.044715 * {a} * {a} * {a})) - 1.0)"


OP_TABLE: dict[str, OpDef] = {
    "add": OpDef(2, lambda a: f"({a[0]} + {a[1]})", lambda g, a, o: [g, g], torch.add),
    "sub": OpDef(2, lambda a: f"({a[0]} - {a[1]})", lambda g, a, o: [g, f"(-({g}))"], torch.sub),
    "mul": OpDef(2, lambda a: f"({a[0]} * {a[1]})", lambda g, a, o: [f"({g} * {a[1]})", f"({g} * {a[0]})"],
                 torch.mul),
    "div": OpDef(2, lambda a: f"({a[0]} / {a[1]})",
                 lambda g, a, o: [f"({g} / {a[1]})", f"(-({g}) * {a[0]} / ({a[1]} * {a[1]}))"], torch.div, True),
    "neg": OpDef(1, lambda a: f"(-({a[0]}))", lambda g, a, o: [f"(-({g}))"], torch.neg),
    "exp": OpDef(1, lambda a: f"tl.exp({a[0]})", lambda g, a, o: [f"({g} * {o})"], torch.exp),
    "log": OpDef(1, lambda a: f"tl.log({a[0]})", lambda g, a, o: [f"({g} / {a[0]})"], torch.log, True),
    "sqrt": OpDef(1, lambda a: f"tl.sqrt({a[0]})", lambda g, a, o: [f"({g} * 0.5 / {o})"], torch.sqrt, True),
    "rsqrt": OpDef(1, lambda a: f"tl.rsqrt({a[0]})", lambda g, a, o: [f"({g} * (-0.5) * {o} / {a[0]})"],
                   torch.rsqrt, True),
    "square": OpDef(1, lambda a: f"({a[0]} * {a[0]})", lambda g, a, o: [f"({g} * 2.0 * {a[0]})"],
                    lambda a: a * a),
    "cube": OpDef(1, lambda a: f"({a[0]} * {a[0]} * {a[0]})",
                  lambda g, a, o: [f"({g} * 3.0 * {a[0]} * {a[0]})"], lambda a: a * a * a),
    "sigmoid": OpDef(1, lambda a: f"tl.sigmoid({a[0]})", lambda g, a, o: [f"({g} * {o} * (1.0 - {o}))"],
                     torch.sigmoid),
    "tanh": OpDef(1, lambda a: f"(2.0 * tl.sigmoid(2.0 * {a[0]}) - 1.0)",
                  lambda g, a, o: [f"({g} * (1.0 - {o} * {o}))"], torch.tanh),
    "silu": OpDef(1, lambda a: f"({a[0]} * tl.sigmoid({a[0]}))",
                  lambda g, a, o: [f"({g} * (tl.sigmoid({a[0]}) * (1.0 + {a[0]} * (1.0 - tl.sigmoid({a[0]})))))"],
                  F.silu),
    "relu": OpDef(1, lambda a: f"tl.maximum({a[0]}, 0.0)",
                  lambda g, a, o: [f"({g} * tl.where({a[0]} > 0.0, 1.0, 0.0))"], F.relu),
    "gelu_tanh": OpDef(
        1, lambda a: f"(0.5 * {a[0]} * (1.0 + {_tanh(a[0])}))",
        lambda g, a, o: [f"({g} * (0.5 * (1.0 + {_tanh(a[0])}) + 0.5 * {a[0]} * (1.0 - {_tanh(a[0])} * "
                         f"{_tanh(a[0])}) * {_C} * (1.0 + 3.0 * 0.044715 * {a[0]} * {a[0]})))"],
        lambda a: F.gelu(a, approximate="tanh")),
    "gelu": OpDef(
        1, lambda a: f"(0.5 * {a[0]} * (1.0 + tl.erf({a[0]} * 0.7071067811865476)))",
        lambda g, a, o: [f"({g} * (0.5 * (1.0 + tl.erf({a[0]} * 0.7071067811865476)) + {a[0]} * "
                         f"0.3989422804014327 * tl.exp(-0.5 * {a[0]} * {a[0]})))"], F.gelu),
    "const": OpDef(0, None, None, None, kind="const"),
    "mean": OpDef(1, None, None, lambda a: a.mean(-1, keepdim=True), kind="reduce"),
    "sum": OpDef(1, None, None, lambda a: a.sum(-1, keepdim=True), kind="reduce"),
}


def _ref(token) -> str:
    return f"x{token[2:]}" if isinstance(token, str) else f"v{token}"


def _validate(chain: list[ChainNode]) -> tuple[int, list[bool]]:
    """Returns (number of external inputs, is_scalar per node)."""
    if not chain:
        raise ValueError("the chain is empty")
    used, scalar = set(), []
    for i, node in enumerate(chain):
        if node.op not in OP_TABLE:
            raise UnsupportedOpError(f"op {node.op!r} is not supported by the generator")
        d = OP_TABLE[node.op]
        if len(node.inputs) != d.arity:
            raise ValueError(f"node {i} ({node.op}) takes {d.arity} inputs, got {len(node.inputs)}")
        if d.kind == "const":
            if "value" not in node.attrs:
                raise ValueError(f"node {i} (const) needs attrs['value']")
            scalar.append(True)
            continue
        flags = []
        for r in node.inputs:
            if isinstance(r, int) and not isinstance(r, bool):
                if not 0 <= r < i:
                    raise ValueError(f"node {i} refers to node {r}, which does not exist before it")
                flags.append(scalar[r])
            elif isinstance(r, str) and r.startswith("in") and r[2:].isdigit():
                used.add(int(r[2:]))
                flags.append(False)
            else:
                raise ValueError(f"node {i} has an invalid input {r!r}")
        scalar.append(True if d.kind == "reduce" else all(flags))
    if scalar[-1]:
        raise ValueError("the chain's result must have the input's shape, not be a scalar")
    n_in = max(used) + 1 if used else 0
    if n_in == 0 or used != set(range(n_in)):
        raise ValueError("external inputs must be named in0, in1, ... without gaps, and at least one is needed")
    return n_in, scalar


def _forward_lines(chain: list[ChainNode], row_mode: bool) -> list[str]:
    lines = []
    for i, node in enumerate(chain):
        d = OP_TABLE[node.op]
        args = [_ref(r) for r in node.inputs]
        if d.kind == "const":
            lines.append(f"v{i} = {float(node.attrs['value'])!r}")
        elif d.kind == "reduce":
            scale = " / N" if node.op == "mean" else ""
            lines.append(f"v{i} = tl.sum(tl.where(m, {args[0]}, 0.0), axis=0){scale}")
        else:
            lines.append(f"v{i} = {d.fwd(args)}")
    return lines


def _backward_lines(chain: list[ChainNode], scalar: list[bool], n_in: int) -> list[str]:
    last = len(chain) - 1
    zero = "tl.zeros([BLOCK], dtype=tl.float32)"
    lines = [f"g{i} = {'0.0' if scalar[i] else zero}" for i in range(len(chain)) if chain[i].op != "const"]
    lines += [f"gx{j} = {zero}" for j in range(n_in)]
    lines.append(f"g{last} = dy")
    for i in range(last, -1, -1):
        node, d = chain[i], OP_TABLE[chain[i].op]
        if d.kind == "const":
            continue
        args = [_ref(r) for r in node.inputs]
        if d.kind == "reduce":
            contribs = [f"g{i} / N" if node.op == "mean" else f"g{i}"]
        else:
            contribs = d.bwd(f"g{i}", args, f"v{i}")
        for r, c in zip(node.inputs, contribs, strict=True):
            if isinstance(r, str):
                lines.append(f"gx{r[2:]} = gx{r[2:]} + {c}")
            elif chain[r].op != "const":
                # a vector gradient flowing into a scalar node (e.g. through a broadcast) is summed over the row
                reduce_it = scalar[r] and not scalar[i]
                lines.append(f"g{r} = g{r} + " + (f"tl.sum(tl.where(m, {c}, 0.0), axis=0)" if reduce_it else c))
    return lines


def _source(chain: list[ChainNode], n_in: int, scalar: list[bool], row_mode: bool) -> str:
    ins = ", ".join(f"x{j}_ptr" for j in range(n_in))
    dins = ", ".join(f"dx{j}_ptr" for j in range(n_in))
    if row_mode:
        head = "    row = tl.program_id(0)\n    offs = tl.arange(0, BLOCK)\n    m = offs < N\n    base = row * N + offs\n"
        sig_extra, mask_args = "N", "N"
    else:
        head = ("    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n    m = offs < n\n"
                "    base = offs\n")
        sig_extra, mask_args = "n", "n"
    loads = "".join(f"    x{j} = tl.load(x{j}_ptr + base, mask=m, other=0.0).to(tl.float32)\n" for j in range(n_in))
    fwd = "".join(f"    {ln}\n" for ln in _forward_lines(chain, row_mode))
    bwd = "".join(f"    {ln}\n" for ln in _backward_lines(chain, scalar, n_in))
    stores = "".join(f"    tl.store(dx{j}_ptr + base, gx{j}.to(dx{j}_ptr.dtype.element_ty), mask=m)\n"
                     for j in range(n_in))
    last = len(chain) - 1
    if row_mode:
        launch_f = ("        rows, ncols = n // out.shape[-1], out.shape[-1]\n"
                    "        _fwd[(rows,)]({args}, out, ncols, BLOCK=triton.next_power_of_2(ncols), "
                    "num_warps=num_warps)\n")
        launch_b = ("        rows, ncols = n // dy.shape[-1], dy.shape[-1]\n"
                    "        _bwd[(rows,)]({args}, dy, {dargs}, ncols, BLOCK=triton.next_power_of_2(ncols), "
                    "num_warps=num_warps)\n")
    else:
        launch_f = "        _fwd[(triton.cdiv(n, block),)]({args}, out, n, BLOCK=block, num_warps=num_warps)\n"
        launch_b = ("        _bwd[(triton.cdiv(n, block),)]({args}, dy, {dargs}, n, BLOCK=block, "
                    "num_warps=num_warps)\n")
    args = ", ".join(f"xs[{j}]" for j in range(n_in))
    dargs = ", ".join(f"dxs[{j}]" for j in range(n_in))
    return f'''"""Generated by tinyforge.kernels.codegen (version {GENERATOR_VERSION}). Chain:
{json.dumps([[n.op, n.inputs, n.attrs] for n in chain])}
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd({ins}, out_ptr, {sig_extra}, BLOCK: tl.constexpr):
{head}{loads}{fwd}    tl.store(out_ptr + base, v{last}.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _bwd({ins}, dy_ptr, {dins}, {sig_extra}, BLOCK: tl.constexpr):
{head}{loads}    dy = tl.load(dy_ptr + base, mask=m, other=0.0).to(tl.float32)
{fwd}{bwd}{stores}

class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, num_warps, *xs):
        if any(x.shape != xs[0].shape for x in xs):
            raise ValueError("all inputs of a generated kernel must have the same shape")
        xs = [x.contiguous() for x in xs]
        out = torch.empty_like(xs[0])
        n = out.numel()
{launch_f.format(args=args)}        ctx.save_for_backward(*xs)
        ctx.cfg = (block, num_warps)
        return out

    @staticmethod
    def backward(ctx, dy):
        xs = ctx.saved_tensors
        block, num_warps = ctx.cfg
        dy = dy.contiguous()
        dxs = [torch.empty_like(x) for x in xs]
        n = dy.numel()
{launch_b.format(args=args, dargs=dargs)}        return (None, None, *dxs)


def fn(*xs, BLOCK=1024, num_warps=4):
    return _Fn.apply(BLOCK, num_warps, *xs)
'''.replace("{mask_args}", mask_args)


@dataclass
class GeneratedKernel:
    chain: list[ChainNode]
    source: str
    path: Path
    hash: str
    row_mode: bool
    n_inputs: int
    kernel: Callable
    reference: Callable

    def spec(self, name: str, example_shape: tuple[int, ...] = (512, 1024)) -> KernelSpec:
        positive = any(OP_TABLE[n.op].positive for n in self.chain)
        n_in = self.n_inputs

        def make_inputs(shape, dtype, device):
            if positive:
                return tuple((torch.rand(shape, device=device) + 0.5).to(dtype) for _ in range(n_in))
            return tuple(torch.randn(shape, device=device).to(dtype) for _ in range(n_in))

        tunables = {"num_warps": [2, 4, 8]} if self.row_mode else {"BLOCK": [512, 1024, 2048], "num_warps": [4, 8]}
        return KernelSpec(name=name, reference=self.reference, kernel=self.kernel,
                          dtypes=(torch.float16, torch.bfloat16, torch.float32),
                          supports=lambda shape: len(shape) >= 1, tunables=tunables, make_inputs=make_inputs,
                          example_shape=example_shape)


def _make_reference(chain: list[ChainNode]) -> Callable:
    def reference(*xs):
        vals: list = []
        for node in chain:
            d = OP_TABLE[node.op]
            if d.kind == "const":
                vals.append(float(node.attrs["value"]))
                continue
            args = [xs[int(r[2:])] if isinstance(r, str) else vals[r] for r in node.inputs]
            vals.append(d.torch_fn(*args))
        return vals[-1]
    return reference


def _generated_dir() -> Path:
    base = Path(os.environ.get("TINYFORGE_CACHE") or Path.home() / ".cache" / "tinyforge")
    return base / "generated"


_LOADED: dict[str, object] = {}


def generate(chain: list[ChainNode]) -> GeneratedKernel:
    n_in, scalar = _validate(chain)
    row_mode = any(OP_TABLE[n.op].kind == "reduce" for n in chain)
    source = _source(chain, n_in, scalar, row_mode)
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    path = _generated_dir() / f"{digest}.py"
    if not path.exists() or path.read_text(encoding="utf-8") != source:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(source, encoding="utf-8")
        os.replace(tmp, path)
    key = f"{path}"
    if key not in _LOADED:
        spec = importlib.util.spec_from_file_location(f"tinyforge_generated_{digest}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED[key] = module
    return GeneratedKernel(chain, source, path, digest, row_mode, n_in, _LOADED[key].fn,
                           _make_reference(chain))
