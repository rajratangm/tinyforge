"""Static scan of model files for code that would run when they are loaded. Nothing is ever unpickled.

Pickle-based checkpoints (.bin .pt .pth .pkl .ckpt, and the data.pkl inside a torch zip) can execute arbitrary
code on load. This walks the pickle opcode stream and reports every global (module.name) it would import:
  critical  a global that runs code or touches the system (os.system, subprocess.*, builtins.eval, ...)
  warning   a global outside the small allow-list of what real checkpoints use (tensor rebuilds, dtypes)
It also flags custom Python shipped next to a model (`auto_map` / *.py means `trust_remote_code`).

What this is NOT: not a malware scanner and not proof of safety. A clean result means "no suspicious import
seen in the pickle stream", not "the weights are trustworthy"; safetensors and GGUF files hold no code at
all and are the safe formats. Obfuscated or truncated streams are reported as warnings, never as clean.
"""

from __future__ import annotations

import json
import pickletools
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

PICKLE_EXT = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib"}
SAFE_EXT = {".safetensors", ".gguf"}
MAX_PICKLE_BYTES = (
    512 * 1024 * 1024
)  # a pickle stream that big is itself suspicious; scan no further than this

DANGEROUS_MODULES = {
    "os",
    "posix",
    "nt",
    "subprocess",
    "sys",
    "socket",
    "shutil",
    "importlib",
    "runpy",
    "pty",
    "ctypes",
    "pdb",
    "webbrowser",
    "urllib",
    "urllib2",
    "requests",
    "http",
    "ftplib",
    "smtplib",
    "telnetlib",
    "asyncio",
    "signal",
    "multiprocessing",
    "threading",
    "code",
    "codeop",
    "commands",
    "platform",
    "pip",
    "setuptools",
    "tempfile",
}
DANGEROUS_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "input",
    "breakpoint",
}
SAFE_GLOBALS = {  # what a plain torch / numpy / HF checkpoint legitimately imports
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("torch", "Size"),
    ("torch", "device"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_device_tensor_from_numpy"),
    ("torch.nn.parameter", "Parameter"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("_codecs", "encode"),
}


@dataclass
class Finding:
    file: str
    severity: str  # critical | warning | info
    code: str
    message: str


def _is_safe_torch_name(module: str, name: str) -> bool:
    return module == "torch" and (
        name.endswith("Storage") or name in {"float16", "bfloat16", "float32", "int64"}
    )


def _classify(module: str, name: str) -> str | None:
    root = module.split(".")[0]
    if root in DANGEROUS_MODULES or (root in {"builtins", "__builtin__"} and name in DANGEROUS_BUILTINS):
        return "critical"
    if (module, name) in SAFE_GLOBALS or _is_safe_torch_name(module, name):
        return None
    return "warning"


def scan_pickle_bytes(data: bytes, label: str) -> list[Finding]:
    out: list[Finding] = []
    strings: list[
        str
    ] = []  # recent string pushes, for STACK_GLOBAL which takes (module, name) from the stack
    memo: dict[int, str] = {}
    n_memo, prev_str = 0, None  # MEMOIZE numbers every object, so count them all; keep only the strings
    try:
        for op, arg, _pos in pickletools.genops(data):
            if isinstance(arg, str) and op.name in {
                "SHORT_BINUNICODE",
                "BINUNICODE",
                "UNICODE",
                "BINUNICODE8",
            }:
                strings.append(arg)
                cur_str = arg
            else:
                cur_str = None
                if op.name in {"BINGET", "LONG_BINGET", "GET"} and arg in memo:
                    strings.append(memo[arg])
            if op.name == "MEMOIZE":
                if prev_str is not None:
                    memo[n_memo] = prev_str
                n_memo += 1
            elif op.name in {"BINPUT", "LONG_BINPUT", "PUT"} and prev_str is not None:
                memo[arg] = prev_str
            prev_str = cur_str
            module = name = None
            if op.name in {"GLOBAL", "INST"} and isinstance(arg, str):
                module, _, name = arg.replace("\n", " ").partition(" ")
            elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
                module, name = strings[-2], strings[-1]
            if module is not None:
                sev = _classify(module, name or "")
                if sev:
                    out.append(
                        Finding(
                            label,
                            sev,
                            "MS001" if sev == "critical" else "MS002",
                            f"imports {module}.{name} when loaded",
                        )
                    )
    except Exception as err:  # truncated or hand-crafted stream: never call that clean
        out.append(Finding(label, "warning", "MS003", f"pickle stream could not be fully parsed ({err})"))
    return out


def _scan_pickle_file(path: Path) -> list[Finding]:
    if zipfile.is_zipfile(path):  # torch >= 1.6 checkpoints are zips holding data.pkl
        out: list[Finding] = []
        with zipfile.ZipFile(path) as z:
            members = [i for i in z.infolist() if i.filename.endswith(".pkl")]
            if not members:
                return [Finding(str(path), "warning", "MS004", "zip archive without a pickle member")]
            for info in members:
                if info.file_size > MAX_PICKLE_BYTES:
                    out.append(
                        Finding(str(path), "warning", "MS005", f"{info.filename} is over the scan limit")
                    )
                    continue
                out += scan_pickle_bytes(z.read(info), f"{path}:{info.filename}")
        return out
    if path.stat().st_size > MAX_PICKLE_BYTES:
        return [Finding(str(path), "warning", "MS005", "file is over the scan limit; not scanned")]
    return scan_pickle_bytes(path.read_bytes(), str(path))


def scan_path(path: str | Path) -> list[Finding]:
    root = Path(path)
    files = (
        [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    )
    out: list[Finding] = []
    for f in files:
        ext = f.suffix.lower()
        if ext in PICKLE_EXT:
            out.append(
                Finding(str(f), "info", "MS000", "pickle format: can run code on load; prefer safetensors")
            )
            out += _scan_pickle_file(f)
        elif ext == ".py":
            out.append(
                Finding(
                    str(f), "warning", "MS006", "custom Python shipped with the model (trust_remote_code)"
                )
            )
        elif f.name == "config.json":
            try:
                if "auto_map" in json.loads(f.read_text(encoding="utf-8")):
                    out.append(
                        Finding(
                            str(f), "warning", "MS006", "config.json has auto_map: needs trust_remote_code"
                        )
                    )
            except (OSError, ValueError):
                out.append(Finding(str(f), "warning", "MS003", "config.json is not valid JSON"))
        elif ext in SAFE_EXT:
            continue
    return out


def verdict(findings: list[Finding]) -> str:
    sev = {f.severity for f in findings}
    return "unsafe" if "critical" in sev else "review" if "warning" in sev else "no suspicious imports found"


def report(findings: list[Finding]) -> dict:
    return {
        "verdict": verdict(findings),
        "findings": [asdict(f) for f in findings],
        "note": "static check only: does not prove a model is trustworthy",
    }
