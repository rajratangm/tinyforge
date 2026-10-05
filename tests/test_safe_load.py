"""Checkpoints are untrusted input: loading must never execute code embedded in a pickle."""

from __future__ import annotations

import pickle

import pytest
import torch

from tinyforge.train import load_model


class _Payload:
    """Unpickling this would run the callable returned by __reduce__ (here: create a marker file)."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_load_model_rejects_pickled_code(tmp_path):
    marker = tmp_path / "pwned.txt"
    ckpt = tmp_path / "evil.pt"
    torch.save({"model": {}, "model_config": {}, "evil": _Payload(str(marker))}, ckpt,
               pickle_protocol=pickle.DEFAULT_PROTOCOL)

    with pytest.raises(pickle.UnpicklingError):
        load_model(ckpt, "cpu")

    assert not marker.exists(), "unpickling executed attacker-controlled code"
