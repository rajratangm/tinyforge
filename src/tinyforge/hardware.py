"""Hardware probing. Never imports heavy libs at module import time."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import asdict, dataclass

import psutil

from .warnings import Level, Report


@dataclass
class HardwareInfo:
    os: str
    cpu_count: int
    ram_gb: float
    torch_version: str | None
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    triton_available: bool
    jax_available: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def device(self) -> str:
        return "cuda" if self.cuda_available else "cpu"


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def probe() -> HardwareInfo:
    torch_version = None
    cuda = False
    name = None
    vram = 0.0
    cc = None
    bf16 = False
    if _has("torch"):
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        if cuda:
            props = torch.cuda.get_device_properties(0)
            name = props.name
            vram = props.total_memory / 1024**3
            cc = (props.major, props.minor)
            # Native bf16 needs compute capability >= 8.0. Recent torch says True on older GPUs (T4)
            # (it emulates bf16 slowly); those should train in fp16 with loss scaling.
            bf16 = props.major >= 8 and torch.cuda.is_bf16_supported()
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        cpu_count=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        torch_version=torch_version,
        cuda_available=cuda,
        gpu_name=name,
        vram_gb=round(vram, 2),
        compute_capability=cc,
        bf16_supported=bf16,
        triton_available=_has("triton"),
        jax_available=_has("jax"),
    )


def diagnose(hw: HardwareInfo) -> Report:
    r = Report()
    if hw.torch_version is None:
        r.add("HW001", Level.ERROR, "PyTorch is not installed.",
              "pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return r
    if not hw.cuda_available:
        r.add("HW002", Level.WARN,
              "No CUDA GPU detected; training will run on CPU (very slow, tiny models only).",
              "Install a CUDA build of PyTorch and a current NVIDIA driver.")
    else:
        if hw.vram_gb < 6:
            r.add("HW003", Level.INFO,
                  f"{hw.vram_gb:.1f} GB VRAM: use small presets; the planner enables "
                  "gradient checkpointing and smaller batches as needed.")
        if hw.compute_capability and hw.compute_capability < (7, 0):
            r.add("HW004", Level.WARN,
                  f"Compute capability {hw.compute_capability} is old; Triton kernels and "
                  "fast attention are unavailable.", "Use --no-triton.")
        if not hw.bf16_supported:
            r.add("HW005", Level.INFO,
                  "bfloat16 unsupported on this GPU; falling back to fp16 with loss scaling.")
    if not hw.triton_available:
        r.add("HW006", Level.INFO, "Triton not installed; using PyTorch kernels.",
              "pip install 'tinyforge[triton]' (Linux/Windows).")
    if hw.jax_available:
        r.add("HW007", Level.INFO, "JAX detected (experimental backend slot; not used for training yet).")
    if hw.ram_gb < 8:
        r.add("HW008", Level.WARN, f"Only {hw.ram_gb} GB system RAM; keep datasets small.")
    return r
