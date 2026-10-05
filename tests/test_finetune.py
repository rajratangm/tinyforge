import random

from tinyforge import finetune
from tinyforge.finetune import ChatDataset, FTConfig
from tinyforge.ft_data import SECRET, _is_val, normalize

SHAPE = {"params": 409_000_000, "hidden": 960, "layers": 32, "vocab": 49152,
         "linear_dims": [960 + 960] * 4 * 32 + [960 + 2560] * 3 * 32}


def test_normalize_alpaca_and_chat_and_garbage():
    m = normalize({"instruction": "Add", "input": "1+1", "output": "2"})
    assert m[0]["content"] == "Add\n\n1+1" and m[-1]["role"] == "assistant"
    assert normalize({"messages": [{"role": "user", "content": "hi"},
                                   {"role": "assistant", "content": "yo"}]})
    assert normalize({"messages": [{"role": "user", "content": "hi"}]}) is None
    assert normalize({"output": "no prompt"}) is None


def test_split_is_deterministic_and_prompt_keyed():
    assert _is_val("same prompt", 5) == _is_val("same prompt", 5)
    share = sum(_is_val(f"p{i}", 10) for i in range(2000)) / 2000
    assert 0.05 < share < 0.15


def test_secret_detector():
    assert SECRET.search("key sk-" + "a" * 24)
    assert not SECRET.search("nothing to see")


def test_planner_prefers_throughput_and_falls_back_to_4bit():
    c = FTConfig()
    t, rep, _ = finetune.plan(c, vram_gb=4.0, shape=SHAPE)
    assert not rep.has_errors and t.quant == "none" and t.grad_checkpointing
    assert t.batch_size > 4 and t.examples_per_step == 16  # larger micro-batch than requested
    # A 7B-class model cannot be fp16 on 8 GB; it must drop to 4-bit instead of failing.
    big = {**SHAPE, "params": 7_000_000_000, "hidden": 4096, "layers": 32}
    t2, rep2, _ = finetune.plan(c, vram_gb=8.0, shape=big)
    assert t2.quant == "4bit" and any(i.code == "FP003" for i in rep2.items)
    # ...and with a forced fp16 it must report an error rather than crash later.
    _, rep3, _ = finetune.plan(c.model_copy(update={"quant": "none"}), vram_gb=8.0, shape=big)
    assert rep3.has_errors


def test_token_budget_batches_cover_everything_and_respect_budget():
    ds = ChatDataset.__new__(ChatDataset)
    rng = random.Random(0)
    ds.items = [([1] * rng.randint(10, 500), [1]) for _ in range(300)]
    batches = ds.batches(token_budget=1024, rng=rng)
    assert sorted(i for b in batches for i in b) == list(range(300))
    for b in batches:
        longest = max(len(ds.items[i][0]) for i in b)
        assert len(b) == 1 or len(b) * longest <= 1024


def test_memory_error_detection():
    import torch

    assert finetune._is_mem_error(RuntimeError("CUBLAS_STATUS_INTERNAL_ERROR when calling"))
    assert finetune._is_mem_error(torch.OutOfMemoryError("CUDA out of memory"))
    assert not finetune._is_mem_error(RuntimeError("shape mismatch"))
