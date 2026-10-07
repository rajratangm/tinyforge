"""Fine-tuning method catalog and a hardware-aware recommender.

A method is a way of changing a pretrained model: a small adapter (LoRA and variants), the whole model (full
fine-tune), or preference optimisation (DPO/ORPO) on chosen-vs-rejected answers. Which ones are possible depends on
GPU memory, system RAM, the model size, the kind of data you have and which backends are installed.
`recommend_methods` returns every method with an estimated training-memory need, whether it fits (and on which
backend), and the reasons, so the user can choose and the tool can suggest.

Memory figures are ESTIMATES (weights + a per-billion-parameter allowance for adapter and optimizer state + a fixed
activation/context overhead); real use varies with sequence length, batch size and checkpointing. Each method also
says whether it has been verified on real hardware in this project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .memtiers import GB, NF4_BYTES_PER_PARAM, Hierarchy

OVERHEAD_GB = 1.2  # CUDA context + activations at modest sequence length (same allowance as the planner)
ADAPTER_GB_PER_B = 0.10  # adapter + gradients + Adam state, per billion base parameters
# Calibrated on the one measured streaming run (Llama-3.1-8B, 4 GB GPU, Windows): Soup's NF4 store was 3.6 GB
# (0.45 bytes/param, embeddings included) and the job completed while free RAM fell from 6.4 to 0.5 GB, i.e.
# roughly 2 GB of process overhead on top of the store. One data point: treat as an estimate.
STREAM_NF4_BYTES_PER_PARAM = 0.45
STREAM_RAM_OVERHEAD_GB = 2.0
FULL_BYTES_PER_PARAM = 16.0  # fp16 weights + grads + fp32 Adam moments + master copy (mixed precision)


@dataclass(frozen=True)
class Method:
    name: str
    title: str
    family: str  # adapter | full | preference
    data: str  # sft | preference
    summary: str
    weight_bytes: float  # bytes per base parameter held on the GPU while training
    extra_gb_per_b: float  # additional GB per billion params (adapter/optimizer/activations)
    backends: tuple[str, ...]
    verified: str  # what has actually been run in this project
    caveats: tuple[str, ...] = ()
    available: bool = True


CATALOG: dict[str, Method] = {
    m.name: m
    for m in [
        Method(
            "lora",
            "LoRA (fp16/bf16 base)",
            "adapter",
            "sft",
            "Train a small low-rank adapter; the base stays frozen in 16-bit. Best quality per memory when it fits.",
            2.0,
            ADAPTER_GB_PER_B,
            ("native", "soup"),
            "native: 360M (loss), 1.5B; soup: not run in fp16",
        ),
        Method(
            "qlora",
            "QLoRA (4-bit base)",
            "adapter",
            "sft",
            "LoRA on a 4-bit (NF4) frozen base: about a quarter of the memory, slightly lower quality.",
            NF4_BYTES_PER_PARAM,
            ADAPTER_GB_PER_B,
            ("native", "soup"),
            "native: 360M, 1.5B (-4.7% loss vs fp16 on a model that fits); soup: 3B and 8B on a 4 GB GPU",
            ("quantisation costs a little accuracy; the adapter is later served on a quantised base",),
        ),
        Method(
            "dora",
            "DoRA (weight-decomposed LoRA)",
            "adapter",
            "sft",
            "LoRA variant that learns magnitude and direction separately; can help at low rank, a bit slower.",
            2.0,
            ADAPTER_GB_PER_B * 1.2,
            ("native",),
            "see tests; not yet benchmarked against plain LoRA",
            ("gains over LoRA are task-dependent and often small",),
        ),
        Method(
            "rslora",
            "rsLoRA (rank-stabilised LoRA)",
            "adapter",
            "sft",
            "LoRA with a rank-aware scaling that keeps learning stable at higher ranks (r >= 32).",
            2.0,
            ADAPTER_GB_PER_B,
            ("native", "soup"),
            "see tests; not yet benchmarked against plain LoRA",
            ("only matters at higher ranks",),
        ),
        Method(
            "full",
            "Full fine-tuning",
            "full",
            "sft",
            "Update every weight. Most flexible, needs ~16 bytes per parameter, so only small models fit.",
            FULL_BYTES_PER_PARAM,
            0.0,
            (),
            "not implemented",
            ("risks forgetting; the worker rejects method: full today",),
            available=False,
        ),
        Method(
            "dpo",
            "DPO (preference tuning)",
            "preference",
            "preference",
            "Learn from chosen-vs-rejected answers with a LoRA adapter (the base with the adapter off is the "
            "reference model).",
            NF4_BYTES_PER_PARAM,
            ADAPTER_GB_PER_B * 2.5,
            ("soup",),
            "NOT yet run here",
            (
                "needs a preference dataset (prompt, chosen, rejected)",
                "longer sequences: ~2.5x adapter memory",
            ),
        ),
        Method(
            "orpo",
            "ORPO (reference-free preference tuning)",
            "preference",
            "preference",
            "Preference tuning in one stage without a reference model; memory close to plain SFT.",
            NF4_BYTES_PER_PARAM,
            ADAPTER_GB_PER_B * 1.5,
            ("soup",),
            "NOT yet run here",
            ("needs a preference dataset (prompt, chosen, rejected)",),
        ),
    ]
}


@dataclass
class MethodRec:
    name: str
    title: str
    family: str
    fits: bool
    backend: str  # native | soup | none
    need_gb: float
    rank: int = 0
    why: list[str] = field(default_factory=list)
    verified: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def weights_gb(m: Method, params_b: float) -> float:
    """GB the base weights occupy in this method's storage format (params_b is in billions)."""
    return params_b * 1e9 * m.weight_bytes / GB


def need_gb(m: Method, params_b: float) -> float:
    """Estimated GPU memory to train resident: weights + adapter/optimizer allowance + fixed overhead."""
    return weights_gb(m, params_b) + params_b * m.extra_gb_per_b + OVERHEAD_GB


def recommend_methods(
    h: Hierarchy,
    params_b: float,
    data: str = "sft",
    backends: tuple[str, ...] = ("native",),
    prefer: str = "quality",
) -> tuple[list[MethodRec], dict]:
    """Every method with fit/backend/memory/reasons, best first. `backends` = those installed.

    prefer: "quality" (largest-memory adapter that fits resident first) or "fit" (smallest memory first).
    """
    vram = h.vram.capacity_gb if h.vram else 0.0
    usable_vram = vram * 0.9
    free_ram = h.ram.free_gb
    ctx = {"vram_gb": vram, "free_ram_gb": round(free_ram, 1), "backends": list(backends), "data": data}
    recs: list[MethodRec] = []
    for m in CATALOG.values():
        need = round(need_gb(m, params_b), 1)
        r = MethodRec(m.name, m.title, m.family, False, "none", need, verified=m.verified)
        r.why.append(m.summary)
        if not m.available:
            r.why.append("not implemented yet")
        elif m.data != data:
            r.why.append(f"needs {m.data} data, you have {data} data")
        elif need <= usable_vram and "native" in m.backends and "native" in backends:
            r.fits, r.backend = True, "native"
            r.why.append(f"~{need} GB fits resident in {vram:g} GB VRAM (native backend)")
        elif "soup" in m.backends and m.family != "full":
            streams_nf4 = m.weight_bytes == NF4_BYTES_PER_PARAM
            store_gb = params_b * 1e9 * (STREAM_NF4_BYTES_PER_PARAM if streams_nf4 else m.weight_bytes) / GB
            need_ram = store_gb + STREAM_RAM_OVERHEAD_GB
            if need_ram <= free_ram and vram >= 2.0:
                if "soup" in backends:
                    r.fits, r.backend = True, "soup"
                    r.why.append(
                        f"~{need} GB would not fit in {vram:g} GB VRAM, but layer streaming keeps the base "
                        f"in RAM (~{store_gb:.1f} GB store + ~{STREAM_RAM_OVERHEAD_GB:g} GB overhead, "
                        f"{free_ram:.1f} GB free)"
                    )
                    if free_ram < need_ram + 1.0:
                        r.why.append("tight on RAM: close Docker/WSL/browsers first or it will swap")
                else:
                    r.why.append(
                        "fits with layer streaming, but the soup backend is not installed (see `doctor`)"
                    )
            else:
                r.why.append(
                    f"needs ~{need_ram:.1f} GB free RAM for the streamed base; only {free_ram:.1f} GB free"
                )
        else:
            r.why.append(
                f"~{need} GB needed, {vram:g} GB VRAM available (no streaming backend for this method)"
            )
        r.why.extend(m.caveats)
        recs.append(r)
    ok = [r for r in recs if r.fits]
    ok.sort(key=lambda r: r.need_gb, reverse=(prefer == "quality"))
    # variants (dora/rslora) are options on top of lora/qlora, not the default pick
    base = [r for r in ok if r.name in ("lora", "qlora", "dpo", "orpo")]
    rest = [r for r in recs if r not in base]
    ordered = base + [r for r in rest if r.fits] + [r for r in rest if not r.fits]
    for i, r in enumerate(ordered, 1):
        r.rank = i if r.fits else 0
    return ordered, ctx
