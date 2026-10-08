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


def test_turn_terminator_follows_template_not_eos():
    from tinyforge.finetune import turn_terminator

    class Tok:
        eos_token = "<|endoftext|>"

        def get_vocab(self):
            return {"<|im_end|>": 1, "<|endoftext|>": 2}

        def apply_chat_template(self, msgs, tokenize=False):
            return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)

    assert turn_terminator(Tok()) == "<|im_end|>"


def test_eval_recomputes_an_fp16_overflow_in_bf16_instead_of_reporting_nan():
    import torch

    class Out:
        def __init__(self, loss):
            self.loss = loss

    class Overflowy(torch.nn.Module):  # NaN under fp16 autocast, fine otherwise: a bf16-trained model on a T4
        def forward(self, **b):
            bad = torch.get_autocast_dtype("cpu") == torch.float16 and torch.is_autocast_enabled("cpu")
            return Out(torch.tensor(float("nan") if bad else 2.0))

    class DS:
        def __len__(self):
            return 4

        def collate(self, idx, device):
            return {"labels": torch.ones((len(idx), 3), dtype=torch.long)}

    stats: dict = {}
    loss = finetune.eval_loss(Overflowy(), DS(), 4, 2, "cpu", torch.float16, 0, stats)
    assert loss == 2.0 and stats["fp16_retries"] == 2
    # With nothing to fall back on (no fp16), a NaN stays a NaN: no silent hiding.
    class AlwaysNan(Overflowy):
        def forward(self, **b):
            return Out(torch.tensor(float("nan")))

    assert finetune.eval_loss(AlwaysNan(), DS(), 4, 2, "cpu", torch.float16, 0, {}) != finetune.eval_loss(
        Overflowy(), DS(), 4, 2, "cpu", torch.float16, 0, {})


def test_multi_gpu_split_is_complete_balanced_and_identical_on_every_rank():
    class DS:
        items = [([1] * n, [1]) for n in (400, 380, 90, 80, 70, 60, 50, 40, 30, 20)]

    batches = [[0], [1], [2, 3], [4, 5], [6, 7, 8], [9]]
    shares = [finetune._share(batches, DS, r, 2) for r in range(2)]
    assert sorted(i for s in shares for b in s for i in b) == list(range(10))  # nothing lost or duplicated
    assert finetune._share(batches, DS, 0, 1) == batches  # one GPU keeps everything, in order

    def load(share):
        return sum(len(b) * max(len(DS.items[i][0]) for i in b) for b in share)

    assert abs(load(shares[0]) - load(shares[1])) <= 400  # balanced to within the largest single batch
    assert shares == [finetune._share(batches, DS, r, 2) for r in range(2)]  # deterministic
