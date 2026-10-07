import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tinyforge.memory import causal_lm_loss, chunked_ce_loss


def _tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).float()


def _batch(masked_prefix=3):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 101, (3, 12), generator=g)
    lab = ids.clone()
    lab[:, :masked_prefix] = -100
    lab[1, -2:] = -100  # ragged masking
    return {"input_ids": ids, "labels": lab, "attention_mask": torch.ones_like(ids)}


def test_matches_hf_loss_for_any_chunk_size():
    m, b = _tiny(), _batch()
    ref = m(**b).loss
    for chunk in (1, 5, 7, 4096):
        got = causal_lm_loss(m, b, chunk)
        assert torch.allclose(got, ref, atol=1e-5), (chunk, got.item(), ref.item())


def test_gradients_match_hf():
    m, b = _tiny(), _batch()
    m(**b).loss.backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    causal_lm_loss(m, b, chunk=5).backward()
    for n, p in m.named_parameters():
        assert torch.allclose(p.grad, ref[n], atol=1e-5), n


def test_all_masked_batch_gives_zero_loss_and_zero_grad():
    m, b = _tiny(), _batch()
    b["labels"][:] = -100
    loss = causal_lm_loss(m, b)
    assert loss.item() == 0.0
    loss.backward()  # must not raise


def test_chunked_ce_standalone():
    h = torch.randn(2, 6, 8, requires_grad=True)
    head = torch.nn.Linear(8, 17, bias=False)
    y = torch.randint(0, 17, (2, 6))
    ref = torch.nn.functional.cross_entropy(head(h[:, :-1]).reshape(-1, 17), y[:, 1:].reshape(-1))
    assert torch.allclose(chunked_ce_loss(h, head, y, chunk=3), ref, atol=1e-6)
