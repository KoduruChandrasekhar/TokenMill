"""Sampling: greedy, temperature, top-k, top-p, and seeded reproducibility."""

from __future__ import annotations

import torch

from infer.engine.sequence import SamplingParams
from infer.sampling import apply_top_k_top_p, make_generator, sample


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 3.0, 2.0], [5.0, -1.0, 4.9]])
    assert sample(logits, [SamplingParams(), SamplingParams()], [None, None]) == [1, 0]


def test_top_k_keeps_k_largest():
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    out = apply_top_k_top_p(logits, torch.tensor([2]), torch.tensor([1.0]))
    assert torch.isinf(out[0, [0, 3]]).all() and torch.isfinite(out[0, [1, 2]]).all()


def test_top_p_cuts_after_mass_passes_p():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    out = apply_top_k_top_p(probs.log(), torch.tensor([-1]), torch.tensor([0.7]))
    # 0.5 < 0.7 so the second token is still needed; after it the mass is 0.8 > 0.7: cut the rest
    assert torch.isfinite(out[0, :2]).all() and torch.isinf(out[0, 2:]).all()


def test_top_p_keeps_at_least_one():
    out = apply_top_k_top_p(torch.tensor([[10.0, 0.0]]), torch.tensor([-1]), torch.tensor([1e-6]))
    assert torch.isfinite(out[0, 0]) and torch.isinf(out[0, 1])


def test_seeded_sampling_is_reproducible_and_batch_independent():
    torch.manual_seed(0)
    logits = torch.randn(3, 50)
    p = SamplingParams(temperature=0.8, top_k=20, top_p=0.9, seed=1234)
    fb = torch.Generator().manual_seed(0)
    a = [sample(logits[1:2], [p], [make_generator(1234, torch.device("cpu"), fb)])[0] for _ in range(1)]
    g = make_generator(1234, torch.device("cpu"), fb)
    seq_a = [sample(logits[1:2], [p], [g])[0] for _ in range(20)]
    g = make_generator(1234, torch.device("cpu"), fb)
    others = [make_generator(i, torch.device("cpu"), fb) for i in range(2)]
    seq_b = []
    for _ in range(20):                       # same row, but now inside a batch with other requests
        toks = sample(logits, [SamplingParams(temperature=1.0, seed=7), p, SamplingParams()],
                      [others[0], g, None])
        seq_b.append(toks[1])
    assert seq_a == seq_b and a[0] == seq_a[0]


def test_temperature_sharpens():
    logits = torch.tensor([[2.0, 1.0, 0.0]]).repeat(4000, 1)
    fb = torch.Generator().manual_seed(0)

    def frac_top(t):
        params = [SamplingParams(temperature=t)] * len(logits)
        gens = [make_generator(i, torch.device("cpu"), fb) for i in range(len(logits))]
        toks = sample(logits, params, gens)
        return sum(t == 0 for t in toks) / len(toks)

    assert frac_top(0.3) > frac_top(1.0) > 0.4
