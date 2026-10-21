import pytest
import torch

from tinyforge.kernels import env


def test_status_never_raises_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    s = env.triton_status()
    assert s.ok is False
    assert s.reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_smoke_kernel_runs():
    s = env.triton_status()
    assert s.ok is True, s.reason
    assert s.gpu
