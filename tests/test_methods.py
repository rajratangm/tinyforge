"""Fine-tuning method catalog and the hardware-aware recommender (synthetic hardware, no GPU)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tinyforge import methods
from tinyforge.cli import app
from tinyforge.memtiers import Hierarchy, Tier


def hier(vram_gb=4.0, free_ram=6.0) -> Hierarchy:
    vram = Tier("vram", "gpu", vram_gb, vram_gb * 0.8, 150.0, True) if vram_gb else None
    return Hierarchy(
        vram,
        Tier("ram", "ddr4", 16, free_ram, 16.0, True),
        Tier("disk", "nvme", 900, 200, 1.5, True),
        {},
        0.0,
        10.0,
        "test",
    )


def by(recs):
    return {r.name: r for r in recs}


def test_catalog_is_honest_about_what_exists_and_what_was_verified():
    c = methods.CATALOG
    assert not c["full"].available and c["full"].backends == ()
    assert c["dpo"].verified.startswith("NOT") and c["orpo"].verified.startswith("NOT")
    assert "8B" in c["qlora"].verified and all(m.verified for m in c.values())


def test_memory_estimates_use_billions_correctly():
    q = methods.CATALOG["qlora"]
    assert 4.0 < methods.weights_gb(q, 8.0) < 4.8  # 8B x 0.55 bytes
    assert methods.need_gb(methods.CATALOG["lora"], 0.36) < methods.need_gb(methods.CATALOG["lora"], 8.0)
    assert methods.need_gb(methods.CATALOG["full"], 1.0) > 15  # ~16 bytes/param


def test_small_model_fits_resident_and_quality_prefers_fp16_lora():
    recs, _ = methods.recommend_methods(hier(), 0.36, "sft", ("native",), "quality")
    r = by(recs)
    assert r["lora"].fits and r["lora"].backend == "native" and r["qlora"].fits
    assert recs[0].name == "lora"  # most memory-hungry adapter that fits = highest quality
    fit_first, _ = methods.recommend_methods(hier(), 0.36, "sft", ("native",), "fit")
    assert fit_first[0].name == "qlora"


def test_1_5b_on_4gb_needs_4bit():
    recs, _ = methods.recommend_methods(hier(), 1.5, "sft", ("native",))
    r = by(recs)
    assert r["qlora"].fits and not r["lora"].fits


def test_8b_on_4gb_needs_streaming_and_says_so_only_if_the_backend_exists():
    with_soup, _ = methods.recommend_methods(hier(free_ram=6.0), 8.0, "sft", ("native", "soup"))
    r = by(with_soup)
    assert r["qlora"].fits and r["qlora"].backend == "soup" and any("streaming" in w for w in r["qlora"].why)
    assert not r["lora"].fits  # fp16 base would need ~16 GB of RAM
    without, _ = methods.recommend_methods(hier(free_ram=6.0), 8.0, "sft", ("native",))
    r2 = by(without)
    assert not r2["qlora"].fits and any("soup backend is not installed" in w for w in r2["qlora"].why)


def test_tight_ram_warns_and_too_little_ram_rejects():
    tight, _ = methods.recommend_methods(hier(free_ram=5.5), 8.0, "sft", ("native", "soup"))
    assert any("tight on RAM" in w for w in by(tight)["qlora"].why)
    none, _ = methods.recommend_methods(hier(free_ram=3.0), 8.0, "sft", ("native", "soup"))
    assert not by(none)["qlora"].fits and any("free RAM" in w for w in by(none)["qlora"].why)


def test_preference_methods_need_preference_data_and_soup():
    sft, _ = methods.recommend_methods(hier(), 1.5, "sft", ("native", "soup"))
    assert not by(sft)["dpo"].fits and any("needs preference data" in w for w in by(sft)["dpo"].why)
    pref, _ = methods.recommend_methods(hier(), 1.5, "preference", ("native", "soup"))
    assert by(pref)["dpo"].fits and by(pref)["dpo"].backend == "soup" and not by(pref)["qlora"].fits


def test_full_finetune_is_listed_but_never_recommended():
    recs, _ = methods.recommend_methods(hier(vram_gb=80), 0.36, "sft", ("native",))
    full = by(recs)["full"]
    assert not full.fits and any("not implemented" in w for w in full.why)


def test_no_gpu_means_nothing_fits_resident():
    recs, _ = methods.recommend_methods(hier(vram_gb=0), 0.36, "sft", ("native",))
    assert not any(r.fits for r in recs)


def test_cli_list_and_suggest_json():
    r = CliRunner()
    names = [m["name"] for m in json.loads(r.invoke(app, ["methods", "list", "--json"]).stdout)]
    assert {"lora", "qlora", "dora", "rslora", "full", "dpo", "orpo"} <= set(names)
    out = json.loads(
        r.invoke(app, ["methods", "suggest", "--params-b", "1.5", "--no-measure", "--json"]).stdout
    )
    assert {m["name"] for m in out["methods"]} == set(names) and "installed_backends" in out["context"]


@pytest.mark.parametrize("data", ["sft", "preference"])
def test_every_method_gets_a_reason(data):
    recs, _ = methods.recommend_methods(hier(), 3.0, data, ("native", "soup"))
    assert all(r.why for r in recs)
