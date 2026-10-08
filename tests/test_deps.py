"""Friendly failures: missing optional dependencies produce advice, not tracebacks."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tinyforge import cli, deps


def test_explain_missing_maps_modules_to_install_commands():
    msg = deps.explain_missing(ModuleNotFoundError("x", name="torch"))
    assert "pip install torch" in msg and "tinyforge doctor" in msg
    assert "tinyforge[finetune]" in deps.explain_missing(ModuleNotFoundError("x", name="peft.tuners"))
    assert "tinyforge[docs]" in deps.explain_missing(ModuleNotFoundError("x", name="markitdown"))
    assert "tinyforge[bench]" in deps.explain_missing(ModuleNotFoundError("x", name="lm_eval"))
    assert deps.explain_missing(ModuleNotFoundError("x", name="some_random_pkg")) is None


def test_main_prints_advice_and_exits_2_for_a_missing_optional_dependency(monkeypatch, capsys):
    def boom():
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "needs 'torch'" in err and "Fix:" in err and "Traceback" not in err


def test_main_does_not_swallow_unrelated_errors(monkeypatch):
    def boom():
        raise ModuleNotFoundError("No module named 'random_thing'", name="random_thing")

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(ModuleNotFoundError):
        cli.main()


def test_components_report_has_every_component_with_an_install_command():
    comps = deps.check_components()
    assert {c["name"] for c in comps} >= {"torch", "finetune", "docs", "bench", "soup", "llama.cpp"}
    assert all(isinstance(c["installed"], bool) and c["install"] and c["unlocks"] for c in comps)


def test_doctor_json_includes_components():
    out = json.loads(CliRunner().invoke(cli.app, ["doctor", "--json"]).stdout)
    assert out["components"] and {"name", "installed", "install"} <= out["components"][0].keys()


def test_incompatible_torchao_gets_a_fix_not_a_traceback(monkeypatch, capsys):
    def boom():
        raise ImportError(
            "Found an incompatible version of torchao. Found version 0.10.0, "
            "but only versions above 0.16.0 are supported"
        )

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "pip install -U torchao" in err and "pip uninstall -y torchao" in err


def test_unrelated_import_errors_still_raise(monkeypatch):
    def boom():
        raise ImportError("cannot import name 'x'")

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(ImportError):
        cli.main()
