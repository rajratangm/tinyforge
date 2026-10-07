from tinyforge import memtiers as m
from tinyforge.memtiers import Hierarchy, Tier


def hier(
    vram=4.0,
    ram_free=12.0,
    disk_kind="NVMe SSD",
    disk_bw=3.0,
    disk_free=500.0,
    tflops=20.0,
    pcie=12.0,
    ram_bw=40.0,
):
    v = Tier("vram", "gpu", vram, vram, 150.0, True) if vram else None
    return Hierarchy(
        v,
        Tier("ram", "DDR4-3200 x2", 16.0, ram_free, ram_bw, True),
        Tier("disk", disk_kind, 900.0, disk_free, disk_bw, True),
        {"h2d_pinned_gbps": pcie},
        0.0,
        tflops,
        "test",
    )


def test_small_model_stays_in_vram():
    p = m.plan_weights(hier(), params=0.36e9, tokens_per_microbatch=2048)
    assert p.feasible and p.weights_tier == "vram" and not p.stream


def test_8b_nf4_streams_from_ram_and_is_compute_bound_on_slow_gpu():
    p = m.plan_weights(hier(tflops=15.0), params=8e9, tokens_per_microbatch=512)
    assert p.feasible and p.weights_tier == "ram" and p.stream
    assert 4.0 < p.weights_gb < 4.6  # 8e9 * 0.55 B
    assert p.bound_by == "compute" and p.tokens_per_s is not None


def test_fast_gpu_slow_link_is_transfer_bound():
    p = m.plan_weights(hier(tflops=300.0, pcie=4.0), params=8e9, tokens_per_microbatch=512)
    assert p.bound_by == "transfer"
    assert abs(p.seconds_per_microbatch - 3 * p.weights_gb / 4.0) < 1e-6


def test_goes_to_disk_when_ram_too_small():
    p = m.plan_weights(hier(ram_free=5.0), params=8e9, tokens_per_microbatch=512)
    assert p.weights_tier == "disk" and p.transfer_gbps == 3.0
    assert any("RAM tier rejected" in r for r in p.reasons)


def test_hdd_is_flagged_and_far_slower_than_nvme():
    nvme = m.plan_weights(hier(ram_free=5.0, tflops=300.0), params=8e9, tokens_per_microbatch=512)
    hdd = m.plan_weights(
        hier(ram_free=5.0, tflops=300.0, disk_kind="HDD", disk_bw=0.15), params=8e9, tokens_per_microbatch=512
    )
    assert hdd.seconds_per_microbatch > 10 * nvme.seconds_per_microbatch
    assert any("HDD" in r for r in hdd.reasons)


def test_infeasible_when_nothing_holds_weights():
    p = m.plan_weights(hier(ram_free=3.0, disk_free=1.0), params=8e9, tokens_per_microbatch=512)
    assert not p.feasible and p.weights_tier == "none"


def test_no_gpu_probe_gives_transfer_only_lower_bound():
    p = m.plan_weights(hier(tflops=None), params=8e9, tokens_per_microbatch=512)
    assert p.feasible and p.bound_by == "transfer" and any("lower bound" in r for r in p.reasons)


def test_headroom_reserved_for_os():
    # 4.6 GB of weights with 6 GB free RAM: 6 - 2.5 headroom = 3.5 usable -> must not claim the RAM tier
    p = m.plan_weights(hier(ram_free=6.0), params=8e9, tokens_per_microbatch=512)
    assert p.weights_tier == "disk"


def test_model_bytes():
    assert m.model_bytes(1e9, "4bit") == 0.55e9 and m.model_bytes(1e9, "none") == 2e9


def test_pressure_warnings():
    w = m.ram_pressure_warnings(hier(ram_free=2.0, vram=4.0))
    assert any("RAM free" in x for x in w)


def test_disk_measure_runs_and_cleans_up(tmp_path):
    w, r = m.measure_disk_gbps(tmp_path, size_mb=16)
    assert w and r and w > 0 and r > 0
    assert list(tmp_path.iterdir()) == []


def test_disk_kind_never_raises(tmp_path):
    kind, _ = m.disk_kind(tmp_path)
    assert kind in {"NVMe SSD", "SSD", "HDD", "unknown"}


def test_memory_cli_json_no_measure():
    import json

    from typer.testing import CliRunner

    from tinyforge.cli import app

    res = CliRunner().invoke(app, ["memory", "--params-b", "0.5", "--no-measure", "--json"])
    assert res.exit_code in (0, 1)
    out = json.loads(res.stdout)
    assert {"hierarchy", "placement", "warnings"} <= out.keys()


def test_gpu_is_read_from_nvidia_smi_when_pytorch_is_missing(monkeypatch):
    import subprocess

    from tinyforge import memtiers

    monkeypatch.setattr("shutil.which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, "NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096, 3300\n", ""
        ),
    )
    t = memtiers._vram_from_nvidia_smi()
    assert t.capacity_gb == 4.0 and 3.1 < t.free_gb < 3.3 and t.bandwidth_gbps is None and not t.measured
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert memtiers._vram_from_nvidia_smi() is None  # no NVIDIA driver: no GPU, no crash
