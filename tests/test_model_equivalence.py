"""Full-model logits match HuggingFace in fp32, and the cached prefill/decode path matches the
no-cache forward pass position by position."""

from __future__ import annotations

import torch

from infer.kv.context import NoCacheContext
from infer.model.qwen2 import SingleSequenceCache


def _ours_logits(model, ids: list[int]) -> torch.Tensor:
    x = torch.tensor(ids)
    with torch.inference_mode():
        return model.compute_logits(model(x, torch.arange(len(ids)), NoCacheContext.single(len(ids), model.cfg)))


def _prompt(model_pair, n=40, seed=0):
    ours, _ = model_pair
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, min(ours.cfg.vocab_size, 100000) - 3, (n,), generator=g).tolist()


def test_full_logits_match_hf(model_pair):
    ours, hf = model_pair
    ids = _prompt(model_pair)
    mine = _ours_logits(ours, ids)
    with torch.inference_mode():
        ref = hf(torch.tensor([ids])).logits[0]
    diff = (mine - ref).abs().max().item()
    assert diff < 5e-4, f"max |logit diff| {diff:.2e}"          # ~1e-4 on the real model, fp32
    assert torch.equal(mine.argmax(-1), ref.argmax(-1))


def test_packed_sequences_are_independent(model_pair):
    """Two sequences side by side in one flat step give the same logits as each alone."""
    ours, _ = model_pair
    a, b = _prompt(model_pair, 17, 1), _prompt(model_pair, 9, 2)
    x = torch.tensor(a + b)
    pos = torch.cat([torch.arange(len(a)), torch.arange(len(b))])
    with torch.inference_mode():
        both = ours.compute_logits(ours(x, pos, NoCacheContext([len(a), len(b)], ours.cfg.head_dim ** -0.5)))
    # Packing changes matmul shapes, so fp32 sums round differently: ~1e-4 on the real model.
    torch.testing.assert_close(both[:len(a)], _ours_logits(ours, a), rtol=1e-3, atol=5e-4)
    torch.testing.assert_close(both[len(a):], _ours_logits(ours, b), rtol=1e-3, atol=5e-4)


def test_kv_cache_decode_matches_full_forward(model_pair):
    """Prefill P tokens, then decode the rest one at a time through the cache: each step's logits
    equal the no-cache forward's logits at that position."""
    ours, _ = model_pair
    ids = _prompt(model_pair, 24, 3)
    full = _ours_logits(ours, ids)
    p = 10
    cache = SingleSequenceCache(ours.cfg, len(ids), torch.float32, torch.device("cpu"))
    with torch.inference_mode():
        h = ours(torch.tensor(ids[:p]), torch.arange(p), cache)
        cache.length = p
        steps = [ours.compute_logits(h[-1:])]
        for t in range(p, len(ids)):
            h = ours(torch.tensor([ids[t]]), torch.tensor([t]), cache)
            cache.length += 1
            steps.append(ours.compute_logits(h))
    got = torch.cat(steps)
    torch.testing.assert_close(got, full[p - 1:], rtol=1e-4, atol=2e-4)
