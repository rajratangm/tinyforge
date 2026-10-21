"""User-defined kernels: loading, scaffolding, the CLI, and the rule that nothing is loaded implicitly."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from tinyforge.cli import app
from tinyforge.kernels import registry, user

TOY = """
import torch
from tinyforge.kernels.registry import KernelSpec

SPEC = KernelSpec(
    name="{name}",
    reference=lambda x: x * 2,
    kernel=lambda x, **t: x * {factor},
    dtypes=(torch.float32,),
    supports=lambda shape: True,
    example_shape=(4, 8),
    make_inputs=lambda shape, dtype, device: (torch.randn(shape, dtype=dtype, device=device),),
)
"""


def write(dir_: Path, fname: str, name="toy", factor="2.0") -> Path:
    p = dir_ / fname
    p.write_text(textwrap.dedent(TOY.format(name=name, factor=factor)), encoding="utf-8")
    return p


def test_load_returns_the_spec_and_keeps_the_global_registry_clean(tmp_path):
    f = write(tmp_path, "a.py", name="user_a")
    for _ in range(2):  # loading twice must not collide
        specs, errors = user.load_user_kernels([f])
        assert [s.name for s in specs] == ["user_a"] and errors == {}
    assert "user_a" not in {s.name for s in registry.all_specs()}


def test_a_directory_loads_every_py_file_in_order(tmp_path):
    write(tmp_path, "b.py", name="user_b")
    write(tmp_path, "a.py", name="user_a")
    (tmp_path / "notes.txt").write_text("ignored")
    specs, errors = user.load_user_kernels([tmp_path])
    assert [s.name for s in specs] == ["user_a", "user_b"] and errors == {}


def test_broken_files_are_reported_not_raised_and_good_files_still_load(tmp_path):
    (tmp_path / "syntax.py").write_text("def (:\n")
    (tmp_path / "nospec.py").write_text("x = 1\n")
    (tmp_path / "wrongtype.py").write_text("SPEC = 3\n")
    write(tmp_path, "good.py", name="user_good")
    specs, errors = user.load_user_kernels([tmp_path])
    assert [s.name for s in specs] == ["user_good"]
    assert set(errors) == {str(tmp_path / n) for n in ("syntax.py", "nospec.py", "wrongtype.py")}
    assert "SPEC" in errors[str(tmp_path / "nospec.py")]


def test_missing_path_is_an_error_entry_not_an_exception(tmp_path):
    specs, errors = user.load_user_kernels([tmp_path / "nope.py"])
    assert specs == [] and len(errors) == 1


def test_search_path_is_explicit_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tinyforge_kernels").mkdir()  # an implicit directory must NOT be picked up
    monkeypatch.delenv("TINYFORGE_KERNEL_PATH", raising=False)
    assert user.kernel_search_path(None) == []
    monkeypatch.setenv("TINYFORGE_KERNEL_PATH", f"{tmp_path / 'x'}{__import__('os').pathsep}{tmp_path / 'y'}")
    assert user.kernel_search_path(None) == [tmp_path / "x", tmp_path / "y"]
    assert user.kernel_search_path(tmp_path / "cli") == [tmp_path / "cli", tmp_path / "x", tmp_path / "y"]


def test_nothing_is_loaded_unless_a_path_is_given(tmp_path, monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from tinyforge.kernels.patch import apply_kernels

    marker = tmp_path / "ran.txt"
    kdir = tmp_path / "tinyforge_kernels"
    kdir.mkdir()
    (kdir / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('x')\nSPEC = None\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TINYFORGE_KERNEL_PATH", raising=False)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    apply_kernels(model, mode="off")
    apply_kernels(model, mode="auto")
    user.kernel_search_path(None)
    assert not marker.exists()


def test_scaffold_writes_a_loadable_template_with_the_given_name(tmp_path):
    p = user.scaffold("my_geglu", tmp_path)
    assert p == tmp_path / "my_geglu.py"
    specs, errors = user.load_user_kernels([p])
    assert errors == {} and specs[0].name == "my_geglu"
    assert specs[0].target is not None and specs[0].example_shape


def test_scaffold_rejects_bad_names_and_never_overwrites(tmp_path):
    for bad in ("", "1abc", "has space", "../x", "a-b"):
        with pytest.raises(ValueError):
            user.scaffold(bad, tmp_path)
    user.scaffold("ok_name", tmp_path)
    with pytest.raises(FileExistsError):
        user.scaffold("ok_name", tmp_path)


def test_cli_new_creates_the_file_and_says_how_to_use_it(tmp_path):
    r = CliRunner().invoke(app, ["kernels", "new", "mine", "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "mine.py").exists()
    assert "--kernel-path" in r.output


def test_cli_verify_file_exits_nonzero_for_a_wrong_kernel(tmp_path):
    f = write(tmp_path, "bad.py", name="user_bad", factor="2.5")
    r = CliRunner().invoke(app, ["kernels", "verify", "--file", str(f), "--device", "cpu", "--dtype", "fp32"])
    assert r.exit_code == 1
    assert "user_bad" in r.output and "verified on" in r.output


def test_cli_verify_file_exits_zero_for_a_correct_kernel(tmp_path):
    f = write(tmp_path, "ok.py", name="user_ok")
    r = CliRunner().invoke(app, ["kernels", "verify", "--file", str(f), "--device", "cpu", "--dtype", "fp32"])
    assert r.exit_code == 0, r.output


def test_cli_verify_file_reports_a_broken_file(tmp_path):
    (tmp_path / "broken.py").write_text("def (:\n")
    r = CliRunner().invoke(app, ["kernels", "verify", "--file", str(tmp_path / "broken.py")])
    assert r.exit_code == 1 and "broken.py" in r.output


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_the_shipped_template_kernel_matches_pytorch():
    from tinyforge.kernels.verify import verify

    p = user.scaffold("geglu_check", Path(__import__("tempfile").mkdtemp()))
    spec = user.load_user_kernels([p])[0][0]
    for dtype in (torch.float32, torch.bfloat16):
        r = verify(spec, spec.example_shape, dtype, "cuda", **{k: v[0] for k, v in spec.tunables.items()})
        assert r.ok, r.reason
