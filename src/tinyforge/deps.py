"""Optional components: what is installed, what each one unlocks, and the exact command to get it.

The core install is deliberately light (no PyTorch). Most commands need more, so instead of a Python
traceback a missing piece produces a plain message with the fix (`explain_missing`), and `tinyforge doctor`
lists every component (`check_components`).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

TORCH_HINT = "pip install torch --index-url https://download.pytorch.org/whl/cu124   (or /cpu for no GPU)"


@dataclass(frozen=True)
class Component:
    name: str
    unlocks: str
    install: str
    modules: tuple[str, ...] = ()  # python modules that must import
    executable: str = ""  # env var naming an executable, or "" if not applicable


COMPONENTS = [
    Component("torch", "training, evaluation, in-process generation", TORCH_HINT, ("torch",)),
    Component(
        "finetune",
        "LoRA/QLoRA fine-tuning (native backend)",
        'pip install "tinyforge[finetune]"',
        ("transformers", "peft", "datasets", "accelerate"),
    ),
    Component(
        "bitsandbytes", "4-bit (QLoRA) loading", 'pip install "tinyforge[finetune]"', ("bitsandbytes",)
    ),
    Component(
        "docs", "PDF/Word/PowerPoint/Excel extraction", 'pip install "tinyforge[docs]"', ("markitdown",)
    ),
    Component(
        "bench",
        "standard benchmarks (MMLU, GSM8K, IFEval...)",
        'pip install "tinyforge[bench]"',
        ("lm_eval",),
    ),
    Component(
        "soup",
        "training models larger than VRAM (backend: soup)",
        "install soup-cli in its OWN venv, then set TINYFORGE_SOUP_BIN",
        executable="TINYFORGE_SOUP_BIN",
    ),
    Component(
        "llama.cpp",
        "GGUF export and the llama.cpp serving engine",
        "download llama.cpp, set TINYFORGE_LLAMA_SERVER and TINYFORGE_LLAMACPP_SRC/BIN",
        executable="TINYFORGE_LLAMA_SERVER",
    ),
]

# python module -> (component, human reason). Used to turn ModuleNotFoundError into advice.
MODULE_HINTS = {
    "torch": "torch",
    "transformers": "finetune",
    "peft": "finetune",
    "datasets": "finetune",
    "accelerate": "finetune",
    "bitsandbytes": "bitsandbytes",
    "markitdown": "docs",
    "lm_eval": "bench",
    "httpx": "finetune",
}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _has_executable(c: Component) -> bool:
    explicit = os.environ.get(c.executable, "").strip()
    if explicit:
        return Path(explicit).exists() or bool(shutil.which(explicit))
    return bool(shutil.which("soup" if c.name == "soup" else "llama-server"))


def check_components() -> list[dict]:
    out = []
    for c in COMPONENTS:
        ok = all(_has_module(m) for m in c.modules) if c.modules else _has_executable(c)
        out.append({**asdict(c), "installed": ok})
    return out


def explain_missing(exc: ModuleNotFoundError) -> str | None:
    """A plain-language message for a missing optional dependency, or None if it is not one of ours."""
    top = (exc.name or "").split(".")[0]
    comp = MODULE_HINTS.get(top)
    if comp is None:
        return None
    c = next(x for x in COMPONENTS if x.name == comp)
    return (
        f"This command needs '{top}', which is not installed ({c.unlocks}).\n"
        f"  Fix: {c.install}\n"
        "  Run `tinyforge doctor` to see everything that is installed or missing."
    )
