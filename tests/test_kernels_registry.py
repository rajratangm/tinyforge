import pytest
import torch

from tinyforge.kernels import registry
from tinyforge.kernels.registry import KernelSpec


def _spec(name: str) -> KernelSpec:
    return KernelSpec(
        name=name,
        reference=lambda x: x * 2,
        kernel=lambda x, **tune: x * 2,
        dtypes=(torch.float32,),
        supports=lambda shape: len(shape) == 2,
        tunables={"num_warps": [2, 4]},
        make_inputs=lambda shape, dtype, device: (torch.ones(shape, dtype=dtype, device=device),),
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    before = dict(registry._SPECS)
    yield
    registry._SPECS.clear()
    registry._SPECS.update(before)


def test_register_and_get():
    s = _spec("toy")
    registry.register(s)
    assert registry.get("toy") is s
    assert s in registry.all_specs()


def test_duplicate_register_raises():
    registry.register(_spec("toy"))
    with pytest.raises(ValueError):
        registry.register(_spec("toy"))


def test_unknown_get_raises():
    with pytest.raises(KeyError):
        registry.get("does-not-exist")
