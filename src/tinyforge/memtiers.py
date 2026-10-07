"""Memory hierarchy: what each level holds, how fast it moves, and where a training job should put things.

Levels (fastest/smallest first): VRAM (GDDR/HBM) -> pinned host RAM (DDR) -> pageable host RAM -> SSD/NVMe -> HDD.
Capacity is not enough to plan with. Layer streaming, activation offload and paged optimisers all turn into
"bytes moved per step / bandwidth of the slowest link in the path", so every tier carries a *measured* bandwidth, and
a cost model compares transfer time with compute time. Vendor numbers (DDR speed, PCIe generation) are recorded only as
context; planning uses measurements, and every figure says whether it was measured or assumed.

Honest limits: file read speed can include the OS page cache (flagged), compute throughput is a short fp16 matmul probe
times an assumed utilisation factor (MFU_ASSUMED, to be calibrated against benchmarks/), and nothing here has run on
Linux yet.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

GB = 1024**3
MFU_ASSUMED = 0.35          # fraction of probed fp16 matmul throughput that a real training step reaches (assumption)
RAM_HEADROOM_GB = 2.5       # kept free for the OS, the desktop, the browser and Docker/WSL on a dev machine
NF4_BYTES_PER_PARAM = 0.55  # NF4 with double quantisation, per the planner's calibration
PASSES_STREAMED_TRAIN = 3   # forward + recompute-in-backward + backward through each streamed layer
DDR_CODES = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 34: "DDR5"}  # SMBIOS memory type codes


@dataclass
class Tier:
    name: str                       # vram | ram | disk
    kind: str                       # e.g. "GDDR6", "DDR4-3200 x2", "NVMe SSD", "HDD"
    capacity_gb: float
    free_gb: float
    bandwidth_gbps: float | None    # measured GB/s toward the next tier up (None = not measured)
    measured: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class Hierarchy:
    vram: Tier | None
    ram: Tier
    disk: Tier
    link: dict                      # PCIe info + measured host<->device bandwidth
    swap_gb: float
    compute_tflops: float | None    # probed fp16 matmul TFLOPS (None without a GPU)
    os: str

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ probing helpers (each degrades to None)

def _ps(cmd: str, timeout: int = 20) -> str | None:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             timeout=timeout)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def disk_kind(path: Path) -> tuple[str, list[str]]:
    """Classify the device holding `path`: 'NVMe SSD', 'SSD', 'HDD' or 'unknown'."""
    notes: list[str] = []
    path = path.resolve()
    if platform.system() == "Windows" and path.drive:
        letter = path.drive[0]
        raw = _ps(f"(Get-Partition -DriveLetter {letter} | Get-Disk | Get-PhysicalDisk | "
                  "Select-Object -First 1 MediaType,BusType | ConvertTo-Json)")
        if raw:
            try:
                d = json.loads(raw)
                media, bus = str(d.get("MediaType", "")), str(d.get("BusType", ""))
                if "NVMe" in bus:
                    return "NVMe SSD", notes
                if media == "HDD":
                    return "HDD", notes
                if media == "SSD":
                    return "SSD", notes
            except json.JSONDecodeError:
                pass
        notes.append("could not classify the drive (virtual disk, USB or permissions)")
    elif platform.system() == "Linux":
        try:
            dev = Path(subprocess.run(["findmnt", "-no", "SOURCE", "-T", str(path)], capture_output=True,
                                      text=True, timeout=10).stdout.strip()).name
            base = dev.rstrip("0123456789") if not dev.startswith("nvme") else dev.split("p")[0]
            rot = Path(f"/sys/block/{base}/queue/rotational")
            if dev.startswith("nvme"):
                return "NVMe SSD", notes
            if rot.exists():
                return ("HDD" if rot.read_text().strip() == "1" else "SSD"), notes
        except (OSError, subprocess.SubprocessError):
            pass
        notes.append("could not classify the drive")
    return "unknown", notes


def ddr_description() -> str | None:
    """e.g. 'DDR4-3200 x2' from SMBIOS (Windows). Context only: planning uses measured bandwidth."""
    if platform.system() != "Windows":
        return None
    raw = _ps("Get-CimInstance Win32_PhysicalMemory | Select-Object Speed,SMBIOSMemoryType | ConvertTo-Json")
    if not raw:
        return None
    try:
        d = json.loads(raw)
        mods = d if isinstance(d, list) else [d]
        gen = DDR_CODES.get(int(mods[0].get("SMBIOSMemoryType", 0)), "DRAM")
        return f"{gen}-{mods[0].get('Speed')} x{len(mods)}"
    except (json.JSONDecodeError, TypeError, ValueError, IndexError):
        return None


def pcie_info() -> dict:
    """Current/max PCIe generation and width from nvidia-smi. 'current' drops at idle, so it is context only."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,"
                              "pcie.link.width.current,pcie.link.width.max", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        gc, gm, wc, wm = (int(x) for x in out.split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    per_lane = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}.get(gm, 0)  # GB/s per lane per direction
    return {"gen_max": gm, "gen_current": gc, "width_max": wm, "width_current": wc,
            "theoretical_gbps_at_max_gen_and_current_width": round(per_lane * wc, 1)}


def measure_ram_copy_gbps(size_mb: int = 256) -> float | None:
    try:
        import numpy as np

        a = np.ones(size_mb * 1024 * 1024 // 8, dtype=np.float64)
        b = np.empty_like(a)
        best = 0.0
        for _ in range(3):
            t = time.perf_counter()
            np.copyto(b, a)
            dt = time.perf_counter() - t
            best = max(best, 2 * a.nbytes / dt / GB)  # read + write traffic
        return round(best, 1)
    except Exception:
        return None


def measure_gpu(size_mb: int = 256) -> dict:
    """Pinned and pageable host->device copy GB/s, device-to-device GB/s, and fp16 matmul TFLOPS."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        n = size_mb * 1024 * 1024
        out: dict = {}
        dev = torch.empty(n, dtype=torch.uint8, device="cuda")
        for label, pinned in (("h2d_pinned_gbps", True), ("h2d_pageable_gbps", False)):
            host = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            dev.copy_(host, non_blocking=pinned)
            torch.cuda.synchronize()
            best = 0.0
            for _ in range(4):
                t = time.perf_counter()
                dev.copy_(host, non_blocking=pinned)
                torch.cuda.synchronize()
                best = max(best, n / (time.perf_counter() - t) / GB)
            out[label] = round(best, 1)
            del host
        dev2 = torch.empty_like(dev)
        torch.cuda.synchronize()
        best = 0.0
        for _ in range(4):
            t = time.perf_counter()
            dev2.copy_(dev)
            torch.cuda.synchronize()
            best = max(best, 2 * n / (time.perf_counter() - t) / GB)
        out["d2d_gbps"] = round(best, 1)
        del dev, dev2
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        for _ in range(3):
            a @ a
        torch.cuda.synchronize()
        t = time.perf_counter()
        reps = 20
        for _ in range(reps):
            a @ a
        torch.cuda.synchronize()
        out["fp16_matmul_tflops"] = round(2 * 2048**3 * reps / (time.perf_counter() - t) / 1e12, 1)
        torch.cuda.empty_cache()
        return out
    except Exception:
        return {}


def measure_disk_gbps(path: Path, size_mb: int = 512) -> tuple[float | None, float | None]:
    """(write, read) GB/s of a scratch file next to `path`. Write is fsynced. Read may be served from the OS page
    cache, so it is an UPPER bound unless the file exceeds free RAM; callers must say so."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        buf = os.urandom(8 * 1024 * 1024)
        with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
            name = f.name
            t = time.perf_counter()
            for _ in range(size_mb // 8):
                f.write(buf)
            f.flush()
            os.fsync(f.fileno())
            w = size_mb * 1024**2 / (time.perf_counter() - t) / GB
        t = time.perf_counter()
        with open(name, "rb") as f:
            while f.read(8 * 1024 * 1024):
                pass
        r = size_mb * 1024**2 / (time.perf_counter() - t) / GB
        os.unlink(name)
        return round(w, 2), round(r, 2)
    except OSError:
        return None, None


# ------------------------------------------------------------------ the hierarchy


def _vram_from_nvidia_smi() -> Tier | None:
    """GPU 0 capacity/free from nvidia-smi. Bandwidth is NOT measured here (needs PyTorch), so it stays None."""
    import shutil

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        name, total, free = (x.strip() for x in out.split(","))
        return Tier("vram", name, round(float(total) / 1024, 2), round(float(free) / 1024, 2), None, False,
                    ["read from nvidia-smi; install PyTorch to measure bandwidth"])
    except Exception:
        return None

def probe_hierarchy(scratch: Path = Path("runs"), measure: bool = True) -> Hierarchy:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    gpu = measure_gpu() if measure else {}
    link = {**pcie_info(), **{k: v for k, v in gpu.items() if k.startswith("h2d")}}

    vram = None
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram = Tier("vram", torch.cuda.get_device_name(0), round(total / GB, 2), round(free / GB, 2),
                        gpu.get("d2d_gbps"), bool(gpu.get("d2d_gbps")),
                        ["capacity includes what the desktop compositor already uses on a laptop"])
    except Exception:
        pass
    if vram is None:  # PyTorch missing (fresh install): read the GPU from nvidia-smi so planning still sees it
        vram = _vram_from_nvidia_smi()

    ram_bw = measure_ram_copy_gbps() if measure else None
    ram = Tier("ram", ddr_description() or "DRAM", round(vm.total / GB, 1), round(vm.available / GB, 1), ram_bw,
               ram_bw is not None,
               ["usable for offload = free minus headroom", f"pagefile/swap in use: {sw.used / GB:.1f} GB"])

    kind, notes = disk_kind(scratch)
    w, r = measure_disk_gbps(scratch) if measure else (None, None)
    usage = psutil.disk_usage(str(scratch.resolve().anchor or scratch.resolve()))
    if r is not None:
        notes.append("read speed may include the OS page cache (upper bound)")
    disk = Tier("disk", kind, round(usage.total / GB, 1), round(usage.free / GB, 1), r, r is not None,
                notes + ([f"sequential write {w} GB/s (fsynced)"] if w is not None else []))
    return Hierarchy(vram, ram, disk, link, round(sw.total / GB, 1),
                     gpu.get("fp16_matmul_tflops"), f"{platform.system()} {platform.release()}")


# ------------------------------------------------------------------ cost model and placement

@dataclass
class Placement:
    feasible: bool
    weights_tier: str               # vram | ram | disk | none
    weights_gb: float
    stream: bool
    transfer_gbps: float | None     # effective bandwidth of the slowest link on the weight path
    seconds_per_microbatch: float | None
    tokens_per_s: float | None
    bound_by: str                   # compute | transfer | n/a
    reasons: list[str]


def model_bytes(params: float, quant: str) -> float:
    return params * {"4bit": NF4_BYTES_PER_PARAM, "none": 2.0}.get(quant, 2.0)


def plan_weights(h: Hierarchy, params: float, tokens_per_microbatch: int, quant: str = "4bit",
                 resident_overhead_gb: float = 1.2) -> Placement:
    """Decide where frozen weights live and what the step costs.

    resident_overhead_gb: LoRA + optimizer + activations + CUDA context that must stay on the GPU regardless.
    """
    why: list[str] = []
    wgb = model_bytes(params, quant) / GB
    tflops = (h.compute_tflops or 0) * MFU_ASSUMED
    flops = 6.0 * params * tokens_per_microbatch  # fwd + recompute + bwd dX; frozen base needs no dW
    compute_s = flops / (tflops * 1e12) if tflops else None
    vram_total = h.vram.capacity_gb if h.vram else 0.0

    if h.vram and wgb + resident_overhead_gb <= vram_total * 0.9:
        why.append(f"weights {wgb:.1f} GB + {resident_overhead_gb:.1f} GB overhead fit in {vram_total:.1f} GB VRAM")
        tps = tokens_per_microbatch / compute_s if compute_s else None
        return Placement(True, "vram", wgb, False, None, compute_s, tps, "compute" if compute_s else "n/a", why)

    pcie = h.link.get("h2d_pinned_gbps") or h.link.get("h2d_pageable_gbps")
    usable_ram = max(0.0, h.ram.free_gb - RAM_HEADROOM_GB)
    candidates = []
    if wgb <= usable_ram and pcie:
        candidates.append(("ram", min(pcie, h.ram.bandwidth_gbps or pcie)))
    elif wgb > usable_ram:
        why.append(f"weights {wgb:.1f} GB exceed usable RAM {usable_ram:.1f} GB "
                   f"(free {h.ram.free_gb:.1f} minus {RAM_HEADROOM_GB} headroom): RAM tier rejected")
    if wgb <= h.disk.free_gb and pcie and h.disk.bandwidth_gbps:
        candidates.append(("disk", min(pcie, h.disk.bandwidth_gbps)))
    if not candidates:
        why.append("no tier can hold the weights; use a smaller model, 4-bit, or free RAM/disk")
        return Placement(False, "none", wgb, False, None, None, None, "n/a", why)

    tier, bw = candidates[0]  # RAM first when it fits: always faster than disk
    if tier == "disk":
        if h.disk.kind == "HDD":
            why.append("weights would live on an HDD: expect minutes per micro-batch; usable only as a last resort")
        else:
            why.append(f"RAM cannot hold the weights; streaming from {h.disk.kind} at <= {bw:.1f} GB/s")
    else:
        why.append(f"weights in pinned RAM, streamed over PCIe at ~{bw:.1f} GB/s")
    transfer_s = PASSES_STREAMED_TRAIN * wgb / bw
    if compute_s is None:
        sec, bound, tps = transfer_s, "transfer", tokens_per_microbatch / transfer_s
        why.append("no compute probe available: ETA is transfer-only (a lower bound)")
    else:
        sec = max(compute_s, transfer_s)  # copies overlap with compute when double-buffered
        bound = "compute" if compute_s >= transfer_s else "transfer"
        tps = tokens_per_microbatch / sec
        why.append(f"per micro-batch: compute {compute_s:.1f}s vs transfer {transfer_s:.1f}s ({bound}-bound)")
    return Placement(True, tier, wgb, True, bw, sec, tps, bound, why)


def ram_pressure_warnings(h: Hierarchy) -> list[str]:
    out = []
    if h.ram.free_gb < RAM_HEADROOM_GB + 2:
        out.append(f"only {h.ram.free_gb:.1f} GB RAM free: close browsers/Docker/WSL before offloaded training; "
                   "oversubscribed RAM pages to disk and slows training by orders of magnitude")
    if h.vram and h.vram.free_gb < 0.85 * h.vram.capacity_gb:
        out.append(f"VRAM is {100 * (1 - h.vram.free_gb / h.vram.capacity_gb):.0f}% used by other processes "
                   "(desktop, browser, other jobs)")
    if platform.system() == "Windows":
        out.append("Windows: the NVIDIA driver may silently spill VRAM into shared system memory (10x slower). "
                   "Set 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback' in the NVIDIA control panel.")
    return out
