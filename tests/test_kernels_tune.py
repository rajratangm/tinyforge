import torch

from tinyforge.kernels import tune as tune_mod
from tinyforge.kernels.registry import KernelSpec
from tinyforge.kernels.tune import tune


def _spec(kernel=None, memory=False) -> KernelSpec:
    return KernelSpec(
        name="toy",
        reference=lambda x: x * 2,
        kernel=kernel or (lambda x, **t: x * 2),
        dtypes=(torch.float32,),
        supports=lambda shape: True,
        tunables={"num_warps": [2, 4, 8]},
        make_inputs=lambda shape, dtype, device: (torch.randn(shape, dtype=dtype, device=device),),
        memory_kernel=memory,
    )


def _timer(table, calls=None):
    def time_fn(fn, label):
        if calls is not None:
            calls.append(label)
        return table[label]
    return time_fn


TIMES = {"ref": 10.0, "num_warps=2": 8.0, "num_warps=4": 5.0, "num_warps=8": 6.0}


def _tune(spec, tmp_path, table=TIMES, calls=None, **kw):
    return tune(spec, (4, 8), torch.float32, device="cpu", cache_dir=tmp_path,
                time_fn=_timer(table, calls), **kw)


def test_picks_fastest_candidate(tmp_path):
    r = _tune(_spec(), tmp_path)
    assert r.params == {"num_warps": 4}
    assert r.kernel_ms == 5.0 and r.ref_ms == 10.0
    assert r.adopted is True


def test_not_adopted_when_slower(tmp_path):
    table = {"ref": 10.0, "num_warps=2": 9.8, "num_warps=4": 10.5, "num_warps=8": 11.0}
    r = _tune(_spec(), tmp_path, table)
    assert r.adopted is False
    assert "faster" in r.reason


def test_memory_kernel_adopted_even_if_slower(tmp_path):
    table = {"ref": 10.0, "num_warps=2": 12.0, "num_warps=4": 13.0, "num_warps=8": 14.0}
    r = _tune(_spec(memory=True), tmp_path, table)
    assert r.adopted is True
    assert r.params == {"num_warps": 2}


def test_cache_hit_skips_timing(tmp_path):
    calls: list[str] = []
    first = _tune(_spec(), tmp_path, calls=calls)
    n = len(calls)
    second = _tune(_spec(), tmp_path, calls=calls)
    assert n > 0 and len(calls) == n
    assert second == first


def test_cache_key_changes_with_gpu_name(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(tune_mod, "_gpu_name", lambda: "gpu-a")
    _tune(_spec(), tmp_path, calls=calls)
    n = len(calls)
    monkeypatch.setattr(tune_mod, "_gpu_name", lambda: "gpu-b")
    _tune(_spec(), tmp_path, calls=calls)
    assert len(calls) > n


def test_failing_verify_candidates_skipped(tmp_path):
    def kernel(x, num_warps=4, **t):
        if num_warps == 2:
            return x * 3  # fastest in the table but wrong
        return x * 2

    r = _tune(_spec(kernel), tmp_path)
    assert r.params == {"num_warps": 4}
    assert r.adopted is True


def test_all_candidates_failing_means_rejected(tmp_path):
    r = _tune(_spec(lambda x, **t: x * 3), tmp_path)
    assert r.adopted is False
    assert "verif" in r.reason


def test_inference_only_kernels_are_timed_forward_only(tmp_path):
    seen, timing = [], []

    def ref(x):
        if timing:  # ignore the verifier's calls; only the timed runner counts
            seen.append(x.requires_grad)
        return x * 2

    def run_and_time(fn, label):
        timing.append(label == "ref")
        if label == "ref":
            fn()
        timing.pop()
        return TIMES[label]

    spec = _spec()
    spec.reference, spec.needs_grad = ref, False
    tune(spec, (4, 8), torch.float32, device="cpu", cache_dir=tmp_path, time_fn=run_and_time)
    assert seen and not any(seen)  # the reference is timed twice (see the warm-up test), always forward-only


def test_reference_is_retimed_after_the_candidates_and_the_faster_reading_wins(tmp_path):
    """A cold GPU clock makes the first measurement slow. Timing the reference only once, first, would make
    every kernel look faster than it is (and the cache would keep that number), so it is retimed."""
    ref_readings = iter([40.0, 10.0])

    def time_fn(fn, label):
        return next(ref_readings) if label == "ref" else {"num_warps=2": 8.0, "num_warps=4": 5.0,
                                                          "num_warps=8": 6.0}[label]

    r = tune(_spec(), (4, 8), torch.float32, device="cpu", cache_dir=tmp_path, time_fn=time_fn)
    assert r.ref_ms == 10.0 and r.kernel_ms == 5.0
