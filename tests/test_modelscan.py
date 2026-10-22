import json
import pickle
import zipfile

import pytest
import torch

from tinyforge import modelscan


class _Boom:  # pickled only to be scanned, never loaded
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


class _Evil:
    def __reduce__(self):
        return (eval, ("1+1",))


def codes(findings, severity):
    return {f.code for f in findings if f.severity == severity}


@pytest.mark.parametrize("protocol", [0, 2, 4, 5])
@pytest.mark.parametrize("payload", [_Boom(), _Evil()])
def test_code_executing_pickles_are_critical_in_every_protocol(payload, protocol):
    found = modelscan.scan_pickle_bytes(pickle.dumps(payload, protocol=protocol), "x")
    assert "MS001" in codes(found, "critical"), found
    assert modelscan.verdict(found) == "unsafe"


def test_a_plain_torch_checkpoint_has_no_critical_or_warning(tmp_path):
    p = tmp_path / "model.bin"
    torch.save({"w": torch.zeros(3, 3), "b": torch.ones(2, dtype=torch.bfloat16)}, p)
    found = modelscan.scan_path(p)
    assert codes(found, "critical") == set() and codes(found, "warning") == set(), found
    assert "MS000" in codes(found, "info")  # still tells you pickle is the unsafe-by-design format


def test_malicious_data_pkl_inside_a_torch_zip_is_found(tmp_path):
    p = tmp_path / "pytorch_model.bin"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(_Boom(), protocol=2))
    assert modelscan.verdict(modelscan.scan_path(p)) == "unsafe"


def test_truncated_pickle_is_never_reported_clean(tmp_path):
    p = tmp_path / "x.pkl"
    p.write_bytes(pickle.dumps({"a": [1, 2, 3]}, protocol=4)[:-6])
    found = modelscan.scan_path(p)
    assert "MS003" in codes(found, "warning") and modelscan.verdict(found) == "review"


def test_safetensors_and_gguf_are_skipped_and_a_clean_folder_says_so(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 8}))
    assert modelscan.scan_path(tmp_path) == []
    assert modelscan.verdict([]) == "no suspicious imports found"


def test_auto_map_and_shipped_python_need_trust_remote_code(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "m.M"}}))
    (tmp_path / "modeling_x.py").write_text("print('hi')\n")
    found = modelscan.scan_path(tmp_path)
    assert [f.code for f in found].count("MS006") == 2 and modelscan.verdict(found) == "review"
