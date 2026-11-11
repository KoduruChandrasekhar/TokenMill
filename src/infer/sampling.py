"""Greedy, temperature, top-k and top-p sampling over a batch of logits.

Every sequence that samples randomly owns a torch.Generator seeded from its request, so its
tokens do not depend on which other requests happen to share the batch.
"""

from __future__ import annotations

import torch


def apply_top_k_top_p(logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """logits: [B, V] fp32 (already divided by temperature); top_k: [B] long (<= 0 or >= V means
    off); top_p: [B] float. Returns logits with excluded tokens set to -inf."""
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    vocab = logits.shape[-1]
    ranks = torch.arange(vocab, device=logits.device)[None, :]
    k = torch.where(top_k <= 0, torch.full_like(top_k, vocab), top_k.clamp(max=vocab))
    remove = ranks >= k[:, None]                                        # keep the k largest
    probs = sorted_logits.masked_fill(remove, float("-inf")).softmax(dim=-1)
    cum = probs.cumsum(dim=-1)
    remove |= (cum - probs) > top_p[:, None]       # cut once the mass before a token passes p
    remove[:, 0] = False                           # always keep at least one token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def sample(logits: torch.Tensor, params: list, generators: list) -> list[int]:
    """logits: [B, V]; params: SamplingParams per row; generators: torch.Generator or None per row."""
    greedy_tokens = logits.argmax(dim=-1)
    random_rows = [i for i, p in enumerate(params) if not p.greedy]
    if not random_rows:
        return greedy_tokens.tolist()
    idx = torch.tensor(random_rows, device=logits.device)
    sub = logits[idx].float()
    temps = torch.tensor([params[i].temperature for i in random_rows], device=logits.device)
    sub = sub / temps[:, None]
    top_k = torch.tensor([params[i].top_k for i in random_rows], device=logits.device)
    top_p = torch.tensor([params[i].top_p for i in random_rows], device=logits.device, dtype=torch.float32)
    if (top_k > 0).any() or (top_p < 1.0).any():
        sub = apply_top_k_top_p(sub, top_k, top_p)
    probs = sub.softmax(dim=-1)
    tokens = greedy_tokens.tolist()
    for row, i in enumerate(random_rows):
        tokens[i] = int(torch.multinomial(probs[row], 1, generator=generators[i]))
    return tokens


def make_generator(seed: int | None, device: torch.device, fallback: torch.Generator) -> torch.Generator:
    g = torch.Generator(device=device)
    if seed is None:
        seed = int(torch.randint(0, 2**62, (1,), generator=fallback))
    g.manual_seed(seed)
    return g
