"""Finding fusable op chains in a module with torch.fx (phase 2). CPU only: tracing needs no GPU."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyforge.kernels.codegen import generate
from tinyforge.kernels.fuse import find_fusable_chains


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYFORGE_CACHE", str(tmp_path))


class FastGelu(nn.Module):  # the "gelu_new" formula of GPT-2 style models, written with python scalars
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))


class QuickGelu(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class SiluMul(nn.Module):
    def forward(self, a, b):
        return F.silu(a) * b


class RmsLike(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class SumNorm(nn.Module):
    def forward(self, x):
        return x / x.sum(-1, keepdim=True)


class BroadcastWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(8))

    def forward(self, x):
        return torch.sigmoid(x * self.w) + x  # w broadcasts across rows: not expressible, must not be fused


class Branchy(nn.Module):
    def forward(self, x):
        a = torch.sigmoid(x)
        return a, a * 2.0  # `a` is also an output, so it must stay materialised


class DataDependent(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return torch.sigmoid(x) * x
        return x


class WithMatmul(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return torch.sigmoid(self.lin(x)) * torch.tanh(self.lin(x))


X = torch.randn(4, 8)


def test_finds_the_whole_gelu_chain_and_translation_matches_the_module():
    scan = find_fusable_chains(FastGelu(), (X,))
    assert scan.skipped == "" and len(scan.chains) == 1
    chain = scan.chains[0]
    assert len(chain.externals) == 1 and chain.row_mode is False
    assert torch.allclose(generate(chain.nodes).reference(X), FastGelu()(X), atol=1e-5)


def test_finds_quick_gelu():
    scan = find_fusable_chains(QuickGelu(), (X,))
    assert len(scan.chains) == 1
    assert torch.allclose(generate(scan.chains[0].nodes).reference(X), QuickGelu()(X), atol=1e-6)


def test_row_reduction_chain_is_found_and_marked_row_mode():
    scan = find_fusable_chains(SumNorm(), (X,))
    assert len(scan.chains) == 1 and scan.chains[0].row_mode is True
    assert torch.allclose(generate(scan.chains[0].nodes).reference(X), SumNorm()(X), atol=1e-6)


def test_patterns_already_covered_by_library_kernels_are_left_alone():
    s = find_fusable_chains(SiluMul(), (X, X))
    assert s.chains == [] and s.covered == ["swiglu"]
    r = find_fusable_chains(RmsLike(), (X,))
    assert r.chains == [] and r.covered == ["rmsnorm"]


def test_externals_that_would_need_broadcasting_are_not_fused():
    scan = find_fusable_chains(BroadcastWeight(), (X,))
    # x * w needs the broadcast weight, so it stays outside; its result is just another full-shape input
    assert len(scan.chains) == 1
    chain = scan.chains[0]
    assert chain.ops == ["sigmoid", "add"]
    assert [n.name for n in chain.externals][0].startswith("mul")
    assert all(n.op != "get_attr" for n in chain.externals)


def test_intermediates_used_outside_the_chain_stay_materialised():
    assert find_fusable_chains(Branchy(), (X,)).chains == []


def test_untraceable_module_is_reported_not_raised():
    scan = find_fusable_chains(DataDependent(), (X,))
    assert scan.chains == [] and scan.skipped


def test_matmul_breaks_the_chain():
    scan = find_fusable_chains(WithMatmul(), (X,))
    # sigmoid(.) * tanh(.) is a 3-op chain whose inputs are two matmul outputs of the main shape
    assert len(scan.chains) == 1 and len(scan.chains[0].externals) == 2
    assert sorted(n.op for n in scan.chains[0].nodes) == ["mul", "sigmoid", "tanh"]
