import json

import pytest

from tinyforge.modelcard import render

CFG = {"base_model": "org/base", "data_dir": "", "lora_r": 16, "lora_alpha": 32, "quant": "none", "max_steps": 10,
       "examples_per_step": 4, "lr": 0.0002, "seed": 1}


def _run(tmp_path, with_task=True):
    d = tmp_path / "data"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"source": "org/data", "train": 90, "val": 10, "raw": 100,
                                             "dropped_invalid": 2, "dropped_duplicates": 3, "pii_flagged": 1,
                                             "max_len": 128, "truncated_frac": 0.05}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "ft_config.json").write_text(json.dumps({**CFG, "data_dir": str(d)}))
    (run / "eval.json").write_text(json.dumps({"val_loss_base": 1.0, "val_loss_tuned": 0.5, "improvement_pct": 50.0,
                                               "val_examples": 10, "forgetting_pct": 1.5}))
    if with_task:
        rates = {"strict_em": .1, "lenient_em": .2, "valid": .9, "exec_acc": .3,
                 "ci95": {"lenient_em": [.1, .3], "exec_acc": [.2, .4]}}
        (run / "task_eval.json").write_text(json.dumps({"task": "sql", "n": 10, "tuned": rates,
                                                        "base_raw_prompt": rates, "base_instructed": rates,
                                                        "gain_pts_lenient_em": 12.5}))
    return run


def test_card_contains_numbers_from_files(tmp_path):
    card = render(_run(tmp_path))
    assert "org/base" in card and "org/data" in card
    assert "1.000 (base) -> 0.500 (tuned)" in card
    assert "95% CI 20-40%" in card and "+12.5 points" in card
    assert "NOT verified" in card  # licence is never asserted


def test_card_states_missing_inputs(tmp_path):
    run = _run(tmp_path, with_task=False)
    (run / "eval.json").unlink()
    assert "No evaluation results found" in render(run)


def test_card_rejects_non_run_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        render(tmp_path)
