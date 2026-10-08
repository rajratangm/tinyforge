"""Data-parallel LoRA fine-tuning, exercised for real with 2 CPU processes (gloo backend).

A tiny randomly-initialised Llama and a locally-trained tokenizer keep this offline and fast. The same code
path runs on GPUs with NCCL; that part is checked on real hardware (see notebooks/kaggle_multi_gpu.ipynb).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("peft")
pytest.importorskip("transformers")
import torch  # noqa: E402

from tinyforge import ddp, finetune  # noqa: E402
from tinyforge.finetune import FTConfig  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")


def _tiny_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    specials = ["<|pad|>", "<|end|>", "<|user|>", "<|assistant|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["the cat sat on the mat", "hello world how are you", "two plus two is four"] * 20
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=300, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|end|>", pad_token="<|pad|>")
    tok.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
    LlamaForCausalLM(cfg).save_pretrained(path)


def _write_data(d):
    d.mkdir(parents=True, exist_ok=True)
    facts = [("two plus two", "four"), ("the cat sat on", "the mat"), ("hello", "world how are you")]
    rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            for _ in range(24) for q, a in facts]
    for name, part in (("train.jsonl", rows[:60]), ("val.jsonl", rows[60:])):
        (d / name).write_text("\n".join(json.dumps(r) for r in part), encoding="utf-8")


def _cfg(tmp_path, run: str) -> FTConfig:
    return FTConfig(base_model=str(tmp_path / "model"), run_dir=tmp_path / run, data_dir=tmp_path / "data",
                    max_len=64, max_steps=20, batch_size=4, grad_accum=2, lr=2e-3, warmup_steps=2, lora_r=4,
                    lora_alpha=8, quant="none", auto_plan=False, eval_interval=10, eval_examples=8,
                    chunked_ce=False, seed=1)


def _evals(run_dir):
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x)["val_loss"] for x in lines if json.loads(x)["event"] == "eval"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ddp")
    _tiny_model(tmp / "model")
    _write_data(tmp / "data")
    return tmp


def test_single_process_baseline_learns(tiny):
    s = finetune.train(_cfg(tiny, "single"))
    ev = _evals(tiny / "single")
    assert s["world_size"] == 1 and "ddp_max_param_divergence" not in s
    assert ev[-1] < ev[0]


def test_two_process_training_keeps_replicas_identical_and_learns(tiny, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")  # children see no GPU: CPU + gloo, even on a GPU machine
    events: list[dict] = []
    s = ddp.launch(_cfg(tiny, "ddp"), 2, events.append)
    assert s["world_size"] == 2
    assert s["ddp_max_param_divergence"] == 0.0  # every rank holds exactly rank 0's LoRA weights
    ev = _evals(tiny / "ddp")
    assert ev[-1] < ev[0]  # it learns
    assert (tiny / "ddp" / "best" / "adapter_model.safetensors").exists()  # lead rank saved the adapter
    assert any(e["event"] == "step" for e in events) and events[-1]["event"] == "finished"
    # Only the lead rank reports: the metrics log must not contain duplicated step numbers.
    steps = [json.loads(x)["step"] for x in (tiny / "ddp" / "metrics.jsonl").read_text().splitlines()
             if json.loads(x)["event"] == "step"]
    assert steps == sorted(set(steps))


def test_two_process_loss_tracks_single_process(tiny):
    a, b = _evals(tiny / "single")[-1], _evals(tiny / "ddp")[-1]
    # Same hyperparameters and global batch, different sampling order: final held-out loss is close.
    assert abs(a - b) / a < 0.25


def test_launch_reports_a_failing_worker(tiny, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    bad = _cfg(tiny, "bad").model_copy(update={"base_model": str(tiny / "does-not-exist")})
    with pytest.raises(RuntimeError):
        ddp.launch(bad, 2)
