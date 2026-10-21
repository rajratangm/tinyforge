"""Find chains of fusable ops in a module with torch.fx, for the code generator in codegen.py.

A chain is a group of ops the generator can express (elementwise math, plus mean/sum over the last dimension)
whose intermediate values are used only inside the group, and whose other inputs all have the chain's shape.
Chains the library already covers by hand-written kernels (silu*up, RMSNorm) are reported, not returned.
A module fx cannot trace (data-dependent control flow, unusual Python) is reported in `skipped`, never raised.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

from .codegen import OP_TABLE, ChainNode

_BINARY = {operator.add: "add", torch.add: "add", operator.sub: "sub", torch.sub: "sub",
           operator.mul: "mul", torch.mul: "mul", operator.truediv: "div", torch.div: "div"}
_UNARY_FN = {torch.neg: "neg", operator.neg: "neg", torch.exp: "exp", torch.log: "log", torch.sqrt: "sqrt",
             torch.rsqrt: "rsqrt", torch.sigmoid: "sigmoid", F.sigmoid: "sigmoid", torch.tanh: "tanh",
             F.tanh: "tanh", F.silu: "silu", F.relu: "relu", torch.relu: "relu", torch.square: "square"}
_UNARY_METHOD = {"neg": "neg", "exp": "exp", "log": "log", "sqrt": "sqrt", "rsqrt": "rsqrt",
                 "sigmoid": "sigmoid", "tanh": "tanh", "relu": "relu", "square": "square"}
_POW = {operator.pow, torch.pow}
_REDUCE_FN = {torch.mean: "mean", torch.sum: "sum"}
_REDUCE_METHOD = {"mean": "mean", "sum": "sum"}


@dataclass
class FoundChain:
    nodes: list[ChainNode]  # for codegen.generate(); inputs refer to ints / "in<k>"
    externals: list[fx.Node]  # in0, in1, ... in that order
    output: fx.Node  # the fx node whose value the chain computes
    members: list[fx.Node]  # every fx node the chain replaces
    row_mode: bool  # contains a reduction over the last dimension

    @property
    def ops(self) -> list[str]:
        return [n.op for n in self.nodes if n.op != "const"]


@dataclass
class ChainScan:
    gm: fx.GraphModule | None = None
    chains: list[FoundChain] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)  # chains left to existing library kernels
    skipped: str = ""  # why the module could not be scanned


def _is_number(a) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool)


def _last_dim_keepdim(node: fx.Node, ndim: int, start: int) -> bool:
    """mean/sum is fusable only as `x.mean(-1, keepdim=True)` (dim and keepdim positional or keyword)."""
    args, kw = node.args, node.kwargs
    dim = args[start] if len(args) > start else kw.get("dim")
    keep = args[start + 1] if len(args) > start + 1 else kw.get("keepdim", False)
    return keep is True and dim in (-1, ndim - 1) and not isinstance(dim, (list, tuple))


def _map(node: fx.Node):
    """(op, tensor-or-number operands) if the generator can express this fx node, else None."""
    t, a = node.target, node.args
    if node.op == "call_function":
        if t in _BINARY and len(a) == 2:
            return _BINARY[t], list(a)
        if t in _UNARY_FN and len(a) >= 1:
            return _UNARY_FN[t], [a[0]]
        if t in _POW and len(a) == 2 and a[1] in (2, 3) and not isinstance(a[1], bool):
            return ("square" if a[1] == 2 else "cube"), [a[0]]
        if t is F.gelu and len(a) >= 1:
            approx = node.kwargs.get("approximate", a[1] if len(a) > 1 else "none")
            return ("gelu_tanh" if approx == "tanh" else "gelu"), [a[0]]
        if t in _REDUCE_FN and a and isinstance(a[0], fx.Node):
            ndim = len(a[0].meta["tensor_meta"].shape)
            return (_REDUCE_FN[t], [a[0]]) if _last_dim_keepdim(node, ndim, 1) else None
    elif node.op == "call_method" and a and isinstance(a[0], fx.Node):
        if t in _UNARY_METHOD:
            return _UNARY_METHOD[t], [a[0]]
        if t == "pow" and len(a) == 2 and a[1] in (2, 3) and not isinstance(a[1], bool):
            return ("square" if a[1] == 2 else "cube"), [a[0]]
        if t in _REDUCE_METHOD:
            ndim = len(a[0].meta["tensor_meta"].shape)
            return (_REDUCE_METHOD[t], [a[0]]) if _last_dim_keepdim(node, ndim, 1) else None
    return None


def _shape(n: fx.Node):
    meta = n.meta.get("tensor_meta")
    return tuple(meta.shape) if meta is not None and hasattr(meta, "shape") else None


def _is_float(n: fx.Node) -> bool:
    meta = n.meta.get("tensor_meta")
    return meta is not None and getattr(meta, "dtype", None) is not None and meta.dtype.is_floating_point


def _grow(root: fx.Node, mapped: dict, assigned: set) -> set:
    main = _shape(root)
    reduced = main[:-1] + (1,)
    chain = {root}
    while True:
        added = False
        for n in list(chain):
            for p in mapped[n][1]:
                if (isinstance(p, fx.Node) and p in mapped and p not in chain and p not in assigned
                        and all(u in chain for u in p.users) and _shape(p) in (main, reduced)):
                    chain.add(p)
                    added = True
        if not added:
            break
    # every input from outside the chain must be a full-shape tensor; shed members that break this
    while chain:
        bad = {n for n in chain for p in mapped[n][1]
               if isinstance(p, fx.Node) and p not in chain and (_shape(p) != main or not _is_float(p))}
        if not bad:
            return chain
        chain -= bad
        # whatever consumed a shed member now reads it from outside: re-check from a consistent core
        chain = {n for n in chain if n is root or any(u in chain for u in n.users)}
    return chain


def _build(chain: set, mapped: dict, order: list) -> FoundChain | None:
    members = [n for n in order if n in chain]
    index: dict[fx.Node, int] = {}
    externals: list[fx.Node] = []
    nodes: list[ChainNode] = []

    def ref(p):
        if isinstance(p, fx.Node) and p in index:
            return index[p]
        if isinstance(p, fx.Node):
            if p not in externals:
                externals.append(p)
            return f"in{externals.index(p)}"
        nodes.append(ChainNode("const", [], {"value": float(p)}))
        return len(nodes) - 1

    for n in members:
        op, operands = mapped[n]
        inputs = [ref(p) for p in operands]
        nodes.append(ChainNode(op, inputs, {}))
        index[n] = len(nodes) - 1
    return FoundChain(nodes, externals, members[-1], members, any(n.op in ("mean", "sum") for n in nodes))


def _covered_by_library(c: FoundChain) -> str | None:
    ops = sorted(c.ops)
    if ops == ["mul", "silu"] and len(c.externals) == 2:
        return "swiglu"
    if "mean" in ops and "rsqrt" in ops:
        return "rmsnorm"
    return None


def find_fusable_chains(module: torch.nn.Module, example_args: tuple) -> ChainScan:
    try:
        gm = fx.symbolic_trace(module)
        ShapeProp(gm).propagate(*example_args)
    except Exception as e:  # untraceable module: report why, never raise
        return ChainScan(skipped=f"{type(e).__name__}: {e}"[:300])
    order = list(gm.graph.nodes)
    mapped: dict[fx.Node, tuple] = {}
    for n in order:
        if n.op in ("call_function", "call_method") and n.meta.get("tensor_meta") is not None:
            try:
                m = _map(n)
            except Exception:
                m = None
            if m is not None and _is_float(n) and all(
                    _is_number(p) or (isinstance(p, fx.Node) and p.meta.get("tensor_meta") is not None)
                    for p in m[1]):
                mapped[n] = m
    scan = ChainScan(gm=gm)
    assigned: set = set()
    for root in reversed(order):
        if root not in mapped or root in assigned or OP_TABLE[mapped[root][0]].kind == "reduce":
            continue  # a chain's result must have the full shape, so it cannot end in a reduction
        chain = _grow(root, mapped, assigned)
        if not chain or root not in chain:
            continue
        found = _build(chain, mapped, order)
        if found is None or len(found.ops) < 2 or not found.externals:
            continue
        assigned |= chain
        covered = _covered_by_library(found)
        if covered:
            scan.covered.append(covered)
        else:
            scan.chains.append(found)
    scan.chains.reverse()  # graph order
    return scan
