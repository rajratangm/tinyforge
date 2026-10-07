from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tinyforge.cli import app
from tinyforge.gguf_export import ExportError, Tools, export_gguf


def _tools(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for n in ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py"):
        (src / n).write_text("")
    q = tmp_path / "llama-quantize"
    q.write_text("")
    return Tools(src / "convert_hf_to_gguf.py", src / "convert_lora_to_gguf.py", src / "gguf-py", q)


def _dirs(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    ad = tmp_path / "ad"
    ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(b"x")
    return base, ad


def test_steps_order_flags_and_cleanup(tmp_path):
    base, ad = _dirs(tmp_path)
    calls = []

    def fake_run(cmd, env, log):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--outfile") + 1]) if "--outfile" in cmd else Path(cmd[2])
        out.write_bytes(b"gguf")
        assert "gguf-py" in env["PYTHONPATH"] or cmd[0].endswith("llama-quantize")

    res = export_gguf(base, tmp_path / "out", ad, "q4_k_m", tools=_tools(tmp_path), run=fake_run)
    names = [Path(c[1]).name if c[0] == sys.executable else Path(c[0]).name for c in calls]
    assert names == ["convert_lora_to_gguf.py", "convert_hf_to_gguf.py", "llama-quantize"]
    assert calls[2][-1] == "Q4_K_M"
    assert Path(res["base"]).name == "base-q4_k_m.gguf" and Path(res["adapter"]).name == "adapter-f16.gguf"
    assert not (tmp_path / "out" / "base-f16.gguf").exists()  # big intermediate removed


def test_keep_f16_and_f16_only_skip_quantize(tmp_path):
    base, _ = _dirs(tmp_path)
    calls = []

    def fake_run(cmd, env, log):
        calls.append(cmd)
        Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"g")

    res = export_gguf(base, tmp_path / "o", None, "f16", tools=_tools(tmp_path), run=fake_run)
    assert len(calls) == 1 and res["adapter"] is None and Path(res["base"]).name == "base-f16.gguf"


CASES = [("q9", True, True), ("q8_0", False, True), ("q8_0", True, False)]


@pytest.mark.parametrize("quant,adapter_ok,base_ok", CASES)
def test_bad_inputs_fail_before_running_anything(tmp_path, quant, adapter_ok, base_ok):
    base, ad = _dirs(tmp_path)
    if not adapter_ok:
        (ad / "adapter_model.safetensors").unlink()
    if not base_ok:
        (base / "config.json").unlink()

    def boom(*a):
        pytest.fail("must not run")

    with pytest.raises(ExportError):
        export_gguf(base, tmp_path / "o", ad, quant, tools=_tools(tmp_path), run=boom)


def test_failing_tool_reports_the_tail_of_its_output(tmp_path):
    base, ad = _dirs(tmp_path)
    t = _tools(tmp_path)
    script = "import sys\nprint('line1')\nprint('boom: bad tensor', file=sys.stderr)\nsys.exit(3)"
    t.convert_lora.write_text(script)
    with pytest.raises(ExportError, match="convert_lora_to_gguf.py failed.*boom: bad tensor"):
        export_gguf(base, tmp_path / "o", ad, "q8_0", tools=t)


def test_missing_tools_message(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYFORGE_LLAMACPP_SRC", str(tmp_path / "nope"))
    monkeypatch.setenv("TINYFORGE_LLAMACPP_BIN", str(tmp_path / "nobin"))
    with pytest.raises(ExportError, match="TINYFORGE_LLAMACPP_SRC"):
        Tools.locate()


def test_cli_reports_errors_with_exit_code(tmp_path):
    from typer.testing import CliRunner

    r = CliRunner().invoke(app, ["export", "gguf", "--base", str(tmp_path), "--out", str(tmp_path / "o")])
    assert r.exit_code == 1 and "config.json" in r.output
