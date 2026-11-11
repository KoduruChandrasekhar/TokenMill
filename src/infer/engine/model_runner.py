"""Turns a scheduler plan into tensors, runs ONE forward pass, and samples one token per sequence.

A step's tokens are laid out flat: every prefill sequence's uncached tokens first, then one token
per decoding sequence. positions are each token's absolute position in its own sequence (never a
shared counter), and slot_mapping says where its K and V go in the cache.
"""

from __future__ import annotations

import math

import torch

from infer.engine.sequence import Sequence
from infer.kv.context import NoCacheContext, PrefillSegment, StepContext
from infer.model.qwen2 import Qwen2ForCausalLM
from infer.sampling import sample


class ModelRunner:
    def __init__(self, model: Qwen2ForCausalLM, cache, device: torch.device, max_model_len: int):
        self.model = model
        self.cache = cache                       # BlockManager or SlotManager (has .pool)
        self.device = device
        self.max_model_len = max_model_len
        self.scale = model.cfg.head_dim ** -0.5
        self.pin = device.type == "cuda"
        self.graphs = None                       # CudaGraphRunner when --cuda-graphs

    def _to_device(self, data: list, dtype=torch.long) -> torch.Tensor:
        t = torch.tensor(data, dtype=dtype, pin_memory=self.pin)
        return t.to(self.device, non_blocking=True)

    def _tables(self, seqs: list[Sequence], max_len: int) -> torch.Tensor:
        bs = self.cache.block_size
        width = max(1, math.ceil(max_len / bs))
        return self._to_device([s.block_table[:width] + [0] * (width - min(width, len(s.block_table)))
                                for s in seqs])

    def prepare(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence],
                chunks: dict | None = None):
        """chunks: tokens to prefill this step per sequence (chunked prefill); default = all uncached."""
        ids, pos, slots, last_idx = [], [], [], []
        segments: list[PrefillSegment] = []
        slot = self.cache.slot
        for seq in prefill_seqs:
            start = seq.num_computed_tokens
            end = start + chunks[seq] if chunks and seq in chunks else seq.length
            tokens = seq.all_token_ids[start:end]
            table = None
            if start > 0:                   # prefix-cache hit or a later chunk: attend over cached blocks
                table = self._tables([seq], end)
            segments.append(PrefillSegment(len(ids), end - start, start, table))
            ids += tokens
            pos += range(start, end)
            slots += [slot(seq, p) for p in range(start, end)]
            last_idx.append(len(ids) - 1)
        decode_start = len(ids)
        ctx_lens = []
        for seq in decode_seqs:
            p = seq.length - 1
            ids.append(seq.last_token)
            pos.append(p)
            slots.append(slot(seq, p))
            ctx_lens.append(seq.length)
            last_idx.append(len(ids) - 1)
        ctx = StepContext(self.cache.pool, self._to_device(slots), self.scale, segments, decode_start,
                          len(decode_seqs))
        if decode_seqs:
            max_len = max(ctx_lens)
            ctx.decode_tables = self._tables(decode_seqs, max_len)
            ctx.decode_context_lens = self._to_device(ctx_lens)
            ctx.decode_max_len = max_len
        return self._to_device(ids), self._to_device(pos), ctx, self._to_device(last_idx)

    @torch.inference_mode()
    def execute(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence],
                chunks: dict | None = None) -> tuple[list[Sequence], list[int]]:
        """One forward pass. Returns the sequences that produced a token and the tokens: every
        decode, and every prefill whose last prompt token was in this step (a partial chunk of a
        longer prompt produces nothing yet)."""
        if not prefill_seqs and self.graphs is not None and self.graphs.can_run(decode_seqs):
            seqs = decode_seqs
            logits = self.graphs.run(decode_seqs)
        else:
            rows = prefill_seqs + decode_seqs
            done = [not chunks or s not in chunks or s.num_computed_tokens + chunks[s] == s.length
                    for s in prefill_seqs] + [True] * len(decode_seqs)
            seqs = [s for s, d in zip(rows, done) if d]
            input_ids, positions, ctx, last_idx = self.prepare(prefill_seqs, decode_seqs, chunks)
            hidden = self.model(input_ids, positions, ctx)
            if not seqs:
                return [], []
            if not all(done):
                last_idx = last_idx[torch.tensor(done, device=last_idx.device)]
            logits = self.model.compute_logits(hidden[last_idx])        # only rows we sample from
        return seqs, sample(logits, [s.params for s in seqs], [s.generator for s in seqs])

    @torch.inference_mode()
    def profile_peak_memory(self, max_num_batched_tokens: int, max_num_seqs: int) -> int:
        """Run one dummy forward at the largest step we will ever schedule and return the peak
        bytes the CUDA caching allocator needed (weights included)."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        n = max_num_batched_tokens
        seg = min(n, self.max_model_len)
        segs = [seg] * (n // seg) + ([n % seg] if n % seg else [])
        ids = torch.zeros(n, dtype=torch.long, device=self.device)
        pos = torch.cat([torch.arange(s, device=self.device) for s in segs])
        hidden = self.model(ids, pos, NoCacheContext(segs, self.scale))
        rows = torch.arange(min(max_num_seqs, n), device=self.device)
        logits = self.model.compute_logits(hidden[rows]).float()
        logits.softmax(-1)
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        del hidden, logits
        torch.cuda.empty_cache()
        return peak
