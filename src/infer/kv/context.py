"""Attention contexts: per-step metadata telling attention which tokens belong to which sequence.

The model calls ctx.attend(layer, q, k, v) once per layer with every token of the step side by
side. A context writes the new K/V into the cache (one batched index_copy_ through the slot
mapping) and then runs attention for each sequence over its own cached context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from infer.model.layers import causal_attention, decode_attention, masked_attention


class NoCacheContext:
    """Causal attention over fresh tokens only (no KV cache). Used for full-sequence logits in
    tests and for the activation-memory profiling pass. seg_lens splits the flat tokens into
    independent sequences."""

    def __init__(self, seg_lens: list[int], scale: float):
        self.seg_lens = seg_lens
        self.scale = scale

    @classmethod
    def single(cls, length: int, cfg) -> NoCacheContext:
        return cls([length], cfg.head_dim ** -0.5)

    def attend(self, layer_idx, q, k, v):
        outs, start = [], 0
        for n in self.seg_lens:
            outs.append(causal_attention(q[start:start + n], k[start:start + n], v[start:start + n], self.scale))
            start += n
        return outs[0] if len(outs) == 1 else torch.cat(outs)


@dataclass
class PrefillSegment:
    start: int                       # offset of this sequence's first query token in the step
    length: int                      # query tokens in this step
    num_cached: int                  # tokens of this sequence already in the cache (prefix hits)
    table: torch.Tensor | None = None  # [1, W] block table, only needed when num_cached > 0


@dataclass
class StepContext:
    """A step that mixes prefill segments (any number of tokens each) and single-token decodes.

    Prefill tokens come first in the flat step, then one token per decoding sequence.
    """

    pool: object                                  # KVPool: write(...) and gather(...)
    slot_mapping: torch.Tensor                    # [T] flat cache slot for every token in the step
    scale: float
    prefills: list[PrefillSegment] = field(default_factory=list)
    decode_start: int = 0
    num_decode: int = 0
    decode_tables: torch.Tensor | None = None     # [B, W] block tables (or [B, 1] slots)
    decode_context_lens: torch.Tensor | None = None  # [B] tokens to attend over, new token included
    decode_max_len: int = 0

    def attend(self, layer_idx, q, k, v):
        self.pool.write(layer_idx, self.slot_mapping, k, v)
        outs = []
        for seg in self.prefills:
            sl = slice(seg.start, seg.start + seg.length)
            if seg.num_cached == 0:
                outs.append(causal_attention(q[sl], k[sl], v[sl], self.scale))
            else:
                ctx_len = seg.num_cached + seg.length
                kc, vc = self.pool.gather(layer_idx, seg.table, ctx_len)          # [1, ctx, KVH, D]
                qpos = seg.num_cached + torch.arange(seg.length, device=q.device)
                mask = torch.arange(ctx_len, device=q.device)[None, :] <= qpos[:, None]
                outs.append(masked_attention(q[sl][None], kc, vc, mask[None], self.scale)[0])
        if self.num_decode:
            kd, vd = self.pool.gather(layer_idx, self.decode_tables, self.decode_max_len)
            qd = q[self.decode_start:self.decode_start + self.num_decode]
            outs.append(decode_attention(qd, kd, vd, self.decode_context_lens, self.scale))
        return outs[0] if len(outs) == 1 else torch.cat(outs)
