"""GGUF export via llama.cpp: LoRA adapter -> GGUF, base model -> GGUF -> quantized.

Serving a LoRA fine-tune with llama.cpp needs two files: the (quantized) base model and the adapter GGUF,
loaded together with `--lora`. This module reproduces the steps verified by hand on an RTX 3050 Ti
(Llama-3.1-8B: adapter f16 + base Q4_K_M, ~18 tok/s) with `tools/llama.cpp` b11380:

    convert_lora_to_gguf.py --base <hf dir> --outtype f16 <adapter dir>   -> adapter-f16.gguf
    convert_hf_to_gguf.py <hf dir> --outtype f16                           -> base-f16.gguf (16 GB for 8B)
    llama-quantize base-f16.gguf base-<quant>.gguf <QUANT>                 -> base-<quant>.gguf

Locations come from `TINYFORGE_LLAMACPP_SRC` (llama.cpp source tree with the convert scripts and gguf-py)
and `TINYFORGE_LLAMACPP_BIN` (folder with llama-quantize / llama-server); both default to `tools/llama.cpp`.

Limits: the base must be a local Hugging Face folder (safetensors); the adapter was trained against an
NF4/fp16 base but is served on a quantized GGUF base, a small mismatch that did not hurt in the one 8B
check; the f16 intermediate needs ~2 bytes per parameter of free disk and is deleted unless keep_f16 is set.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

QUANTS = {"f16", "q8_0", "q4_k_m", "q5_k_m", "q6_k", "q4_0"}
ROOT = Path(__file__).resolve().parents[2]


class ExportError(RuntimeError):
    pass


@dataclass
class Tools:
    convert_hf: Path
    convert_lora: Path
    gguf_py: Path
    quantize: Path

    @classmethod
    def locate(cls) -> Tools:
        default_src = next((ROOT / "tools" / "llama.cpp" / "src").glob("llama.cpp-*"),
                           ROOT / "tools" / "llama.cpp")
        src = Path(os.environ.get("TINYFORGE_LLAMACPP_SRC", "") or default_src)
        binp = Path(os.environ.get("TINYFORGE_LLAMACPP_BIN", "") or ROOT / "tools" / "llama.cpp" / "bin")
        exe = ".exe" if sys.platform == "win32" else ""
        t = cls(src / "convert_hf_to_gguf.py", src / "convert_lora_to_gguf.py", src / "gguf-py",
                binp / f"llama-quantize{exe}")
        missing = [str(p) for p in (t.convert_hf, t.convert_lora, t.quantize) if not p.exists()]
        if missing:
            raise ExportError("llama.cpp tools not found: " + ", ".join(missing) +
                              ". Set TINYFORGE_LLAMACPP_SRC and TINYFORGE_LLAMACPP_BIN.")
        return t


def _run(cmd: list[str], env: dict, log: Callable[[str], None]) -> None:
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-8:]
        raise ExportError(f"{Path(cmd[1] if cmd[0] == sys.executable else cmd[0]).name} failed "
                          f"(exit {p.returncode}): " + " | ".join(tail))


def export_gguf(base_dir: Path, out_dir: Path, adapter_dir: Path | None = None, quant: str = "q4_k_m",
                keep_f16: bool = False, tools: Tools | None = None,
                log: Callable[[str], None] = lambda s: None, run=_run) -> dict:
    """Write `base-<quant>.gguf` (and `adapter-f16.gguf` when an adapter is given) into out_dir."""
    quant = quant.lower()
    if quant not in QUANTS:
        raise ExportError(f"unknown quant {quant!r}; choose one of {sorted(QUANTS)}")
    base_dir, out_dir = Path(base_dir).resolve(), Path(out_dir).resolve()
    if not (base_dir / "config.json").exists():
        raise ExportError(f"{base_dir} is not a Hugging Face model folder (no config.json)")
    if adapter_dir is not None and not (Path(adapter_dir) / "adapter_model.safetensors").exists():
        raise ExportError(f"{adapter_dir} has no adapter_model.safetensors")
    t = tools or Tools.locate()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(t.gguf_py) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "PYTHONUTF8": "1"}
    result: dict = {"base": None, "adapter": None, "quant": quant}
    if adapter_dir is not None:
        adapter_out = out_dir / "adapter-f16.gguf"
        run([sys.executable, str(t.convert_lora), "--base", str(base_dir), "--outtype", "f16",
             "--outfile", str(adapter_out), str(Path(adapter_dir).resolve())], env, log)
        result["adapter"] = str(adapter_out)
    f16 = out_dir / "base-f16.gguf"
    run([sys.executable, str(t.convert_hf), str(base_dir), "--outtype", "f16", "--outfile", str(f16)],
        env, log)
    if quant == "f16":
        result["base"] = str(f16)
        return result
    final = out_dir / f"base-{quant}.gguf"
    run([str(t.quantize), str(f16), str(final), quant.upper()], env, log)
    if not keep_f16:
        f16.unlink(missing_ok=True)
    result["base"] = str(final)
    return result
