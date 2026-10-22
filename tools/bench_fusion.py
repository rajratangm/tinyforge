"""Benchmark generated (auto-fused) kernels against eager PyTorch on real activation modules.

    .venv\\Scripts\\python tools/bench_fusion.py benchmarks/kernels-phase2.json

For each activation module it scans the module with torch.fx, generates the kernel, checks it against the
module (outputs and gradients), tunes it, and times forward+backward and measures peak memory against the
eager module. Run it on an idle GPU.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

from tinyforge.kernels.codegen import generate
from tinyforge.kernels.fuse import find_fusable_chains
from tinyforge.kernels.tune import tune
from tinyforge.kernels.verify import verify


def peak_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def main(out: Path) -> None:
    import os

    os.environ.setdefault("TINYFORGE_CACHE", tempfile.mkdtemp())  # fresh tuning, nothing reused from before
    from transformers.activations import ACT2FN

    names = ["gelu_new", "gelu_fast", "quick_gelu", "mish", "relu2"]
    shape, dtype = (4096, 3072), torch.bfloat16
    results = {"gpu": torch.cuda.get_device_name(0), "shape": list(shape), "dtype": "bfloat16", "modules": {}}
    for name in names:
        module = ACT2FN[name]
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        scan = find_fusable_chains(module, (x,))
        entry: dict = {"class": type(module).__name__}
        if scan.skipped or not scan.chains:
            entry["result"] = f"not fused: {scan.skipped or 'no chain of 2+ ops'}"
            results["modules"][name] = entry
            continue
        chain = scan.chains[0]
        gen = generate(chain.nodes)
        spec = gen.spec(f"bench_{name}", shape)
        tuned = tune(spec, shape, dtype, "cuda", margin=0.0)
        v = verify(spec, shape, dtype, "cuda", **tuned.params)
        xa = x.clone().requires_grad_()
        xb = x.clone().requires_grad_()

        def eager(module=module, xa=xa):
            module(xa).sum().backward()

        params = dict(tuned.params)

        def fused(gen=gen, xb=xb, params=params):
            gen.kernel(xb, **params).sum().backward()

        entry.update({"ops_fused": chain.ops, "verified": v.ok, "tuned_params": tuned.params,
                      "eager_ms": tuned.ref_ms, "fused_ms": tuned.kernel_ms,
                      "speedup": tuned.ref_ms / tuned.kernel_ms if tuned.kernel_ms else None,
                      "eager_peak_mb": peak_mb(eager), "fused_peak_mb": peak_mb(fused)})
        results["modules"][name] = entry
        print(name, {k: entry[k] for k in ("verified", "speedup", "eager_peak_mb", "fused_peak_mb")})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/kernels-phase2.json"))
