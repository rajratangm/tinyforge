"""Triton code generator for elementwise / row-reduction op chains (phase 2)."""

from __future__ import annotations

import pytest
import torch

from tinyforge.kernels import codegen
from tinyforge.kernels.codegen import ChainNode, UnsupportedOpError, generate
from tinyforge.kernels.verify import verify

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYFORGE_CACHE", str(tmp_path))


def N(op, *inputs, **attrs):
    return ChainNode(op, list(inputs), attrs)


SILU_MUL = [N("silu", "in0"), N("mul", 0, "in1")]
# NewGELUActivation from transformers: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
GELU_NEW = [N("cube", "in0"), N("const", value=0.044715), N("mul", 1, 0), N("add", "in0", 2),
            N("const", value=0.7978845608028654), N("mul", 4, 3), N("tanh", 5), N("const", value=1.0),
            N("add", 7, 6), N("const", value=0.5), N("mul", 9, "in0"), N("mul", 10, 8)]
# x * rsqrt(mean(x^2) + eps): a row reduction whose scalar result is broadcast back over the row
RMS_LIKE = [N("square", "in0"), N("mean", 0), N("const", value=1e-6), N("add", 1, 2), N("rsqrt", 3),
            N("mul", "in0", 4)]

ALL_UNARY = ["neg", "exp", "sigmoid", "tanh", "silu", "relu", "gelu_tanh", "gelu", "square", "cube", "log",
             "sqrt", "rsqrt"]
ALL_BINARY = ["add", "sub", "mul", "div"]


def test_op_table_lists_every_supported_op():
    assert set(ALL_UNARY + ALL_BINARY + ["const", "mean", "sum"]) <= set(codegen.OP_TABLE)


def test_unknown_op_raises_unsupported():
    with pytest.raises(UnsupportedOpError, match="matmul"):
        generate([N("matmul", "in0", "in1")])


def test_wrong_arity_and_dangling_references_are_rejected():
    with pytest.raises(ValueError):
        generate([N("add", "in0")])
    with pytest.raises(ValueError):
        generate([N("neg", 5)])  # refers to a node that does not exist
    with pytest.raises(ValueError):
        generate([])


def test_source_is_written_to_the_cache_and_reused_for_the_same_chain(tmp_path):
    a = generate(SILU_MUL)
    assert a.path.exists() and a.path.parent == tmp_path / "generated"
    assert "@triton.jit" in a.path.read_text(encoding="utf-8")
    mtime = a.path.stat().st_mtime_ns
    b = generate([N("silu", "in0"), N("mul", 0, "in1")])  # equal chain, new objects
    assert b.hash == a.hash and b.path == a.path and a.path.stat().st_mtime_ns == mtime
    assert generate(GELU_NEW).hash != a.hash


def test_reference_matches_eager_formula_on_cpu():
    g = generate(GELU_NEW)
    x = torch.randn(4, 8)
    want = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(g.reference(x), want, atol=1e-6)
    r = generate(RMS_LIKE).reference(x)
    assert torch.allclose(r, x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6), atol=1e-6)


def test_row_mode_is_chosen_only_when_the_chain_reduces():
    assert generate(RMS_LIKE).row_mode is True
    assert generate(GELU_NEW).row_mode is False


@cuda
@pytest.mark.parametrize("op", ALL_UNARY)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_every_unary_op_matches_torch_forward_and_backward(op, dtype):
    spec = generate([N(op, "in0")]).spec(f"t_{op}")
    r = verify(spec, (7, 33), dtype, "cuda", **{k: v[0] for k, v in spec.tunables.items()})
    assert r.ok, (op, r.reason)


@cuda
@pytest.mark.parametrize("op", ALL_BINARY)
def test_every_binary_op_matches_torch_forward_and_backward(op):
    spec = generate([N(op, "in0", "in1")]).spec(f"t_{op}")
    r = verify(spec, (7, 33), torch.float32, "cuda", **{k: v[0] for k, v in spec.tunables.items()})
    assert r.ok, (op, r.reason)


@cuda
@pytest.mark.parametrize("chain,shape", [(SILU_MUL, (5, 100)), (GELU_NEW, (3, 1000)), (RMS_LIKE, (6, 960)),
                                         (RMS_LIKE, (2, 1536))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generated_chains_match_eager_with_gradients(chain, shape, dtype):
    spec = generate(chain).spec("t_chain")
    r = verify(spec, shape, dtype, "cuda", **{k: v[0] for k, v in spec.tunables.items()})
    assert r.ok, r.reason


@cuda
def test_row_reduction_with_sum_and_a_scalar_used_twice():
    # y = x / sum(x) + sum(x): the scalar node feeds two consumers, so its gradient must be summed
    chain = [N("sum", "in0"), N("div", "in0", 0), N("add", 1, 0)]
    spec = generate(chain).spec("t_sum")
    r = verify(spec, (4, 50), torch.float32, "cuda", num_warps=4)
    assert r.ok, r.reason


@cuda
def test_a_broken_generator_would_be_caught_by_the_verifier(monkeypatch):
    # sabotage the derivative of silu: the generated kernel then disagrees with autograd, and verify says so
    real = codegen.OP_TABLE["silu"]
    monkeypatch.setitem(codegen.OP_TABLE, "silu", codegen.OpDef(
        real.arity, real.fwd, lambda g, a, o: [f"({g}) * 0.5"], real.torch_fn, real.positive))
    spec = generate([N("silu", "in0")]).spec("t_sabotaged")
    r = verify(spec, (4, 16), torch.float32, "cuda", num_warps=4, BLOCK=512)
    assert r.ok is False and "gradient" in r.reason
