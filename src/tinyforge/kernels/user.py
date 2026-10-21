"""User-defined kernels: load them from files you name, scaffold a template. Never loaded implicitly."""

from __future__ import annotations

import hashlib
import importlib.util
import keyword
import os
import re
from pathlib import Path

from .registry import KernelSpec

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TEMPLATE = Path(__file__).parent / "templates" / "geglu.py"


def kernel_search_path(cli_value: Path | None) -> list[Path]:
    """The files/dirs to load: --kernel-path, then TINYFORGE_KERNEL_PATH. Never the current directory:
    user kernels are arbitrary code, so only paths someone typed are read."""
    paths = [Path(cli_value)] if cli_value else []
    paths += [Path(p) for p in os.environ.get("TINYFORGE_KERNEL_PATH", "").split(os.pathsep) if p]
    return paths


def load_user_kernels(paths: list[Path]) -> tuple[list[KernelSpec], dict[str, str]]:
    """Import each .py file (a directory contributes its *.py, sorted) and collect its SPEC / SPECS.
    Returns (specs, {file: error}); a bad file never stops the others and nothing here raises."""
    specs: list[KernelSpec] = []
    errors: dict[str, str] = {}
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.glob("*.py"))
        elif p.is_file():
            files.append(p)
        else:
            errors[str(p)] = "path does not exist"
    for f in files:
        try:
            digest = hashlib.sha1(str(f.resolve()).encode()).hexdigest()[:10]
            name = f"tinyforge_user_kernel_{digest}_{f.stem}"
            spec = importlib.util.spec_from_file_location(name, f)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "SPECS"):
                found = list(module.SPECS)
            elif hasattr(module, "SPEC"):
                found = [module.SPEC]
            else:
                raise AttributeError("the file defines no SPEC (a KernelSpec) or SPECS (a list of them)")
            for s in found:
                if not isinstance(s, KernelSpec):
                    raise TypeError(f"SPEC must be a KernelSpec, got {type(s).__name__}")
            specs += found
        except (Exception, SystemExit) as e:  # user code: whatever it does, report it and carry on
            errors[str(f)] = f"{type(e).__name__}: {e}"[:300]
    return specs, errors


def scaffold(name: str, out_dir: Path) -> Path:
    """Write a working template (a GeGLU kernel with a model target) to out_dir/NAME.py."""
    if not _NAME.match(name) or keyword.iskeyword(name):
        raise ValueError(f"{name!r} is not a valid kernel name (letters, digits, underscore; not starting "
                         "with a digit)")
    out_dir = Path(out_dir)
    dest = out_dir / f"{name}.py"
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; choose another name or delete it")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(TEMPLATE.read_text(encoding="utf-8").replace("geglu", name), encoding="utf-8")
    return dest
