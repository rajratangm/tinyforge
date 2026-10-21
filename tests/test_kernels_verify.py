import torch

from tinyforge.kernels.registry import KernelSpec
from tinyforge.kernels.verify import verify


def _toy(kernel, name="toy") -> KernelSpec:
    return KernelSpec(
        name=name,
        reference=lambda x: x * 2,
        kernel=kernel,
        dtypes=(torch.float32,),
        supports=lambda shape: True,
        make_inputs=lambda shape, dtype, device: (torch.randn(shape, dtype=dtype, device=device),),
    )


class _WrongGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, g):
        return g * 3  # should be 2


def test_correct_kernel_passes():
    r = verify(_toy(lambda x, **t: x * 2), (4, 8), torch.float32, device="cpu")
    assert r.ok is True
    assert r.max_err == 0 and r.grad_err == 0


def test_wrong_kernel_fails():
    r = verify(_toy(lambda x, **t: x * 2.1), (4, 8), torch.float32, device="cpu")
    assert r.ok is False
    assert r.max_err > 0


def test_wrong_gradient_fails():
    r = verify(_toy(lambda x, **t: _WrongGrad.apply(x)), (4, 8), torch.float32, device="cpu")
    assert r.ok is False
    assert r.max_err == 0
    assert r.grad_err > 0


def test_kernel_exception_is_captured():
    def boom(x, **t):
        raise RuntimeError("compile exploded")

    r = verify(_toy(boom), (4, 8), torch.float32, device="cpu")
    assert r.ok is False
    assert "compile exploded" in r.reason
