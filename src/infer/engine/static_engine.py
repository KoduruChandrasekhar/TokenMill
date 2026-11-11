"""Static batching (baseline B, Part 3), on the same model as the continuous engine.

Collect requests until there are B of them or W ms have passed since the first one arrived.
Left-pad the prompts to one length, mask the padding, give every row its own positions
(mask.cumsum(-1) - 1), prefill the whole batch in one pass, then decode step by step until every
row has finished. Finished rows stay in the batch and keep burning compute; only when all are done
does the next batch start. That idle compute is the waste continuous batching removes.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import torch

from infer.config import EngineConfig
from infer.engine.engine import RequestRejected
from infer.engine.sequence import RequestOutput, SamplingParams, Sequence, Status
from infer.metrics import Metrics
from infer.model.layers import masked_attention
from infer.model.loader import load_model, load_tokenizer, resolve_device, resolve_dtype, resolve_model_path
from infer.sampling import make_generator, sample

log = logging.getLogger("infer.static")


class PaddedContext:
    """Attention for a left-padded batch: cache [layers, 2, B, max_len, KVH, D], one row per sequence."""

    def __init__(self, cache: torch.Tensor, key_mask: torch.Tensor, batch: int, start: int, q_len: int, scale: float):
        self.cache, self.key_mask = cache, key_mask
        self.batch, self.start, self.q_len, self.scale = batch, start, q_len, scale

    def attend(self, layer_idx, q, k, v):
        b, s, n = self.batch, self.start, self.q_len
        q = q.view(b, n, *q.shape[1:])
        self.cache[layer_idx, 0, :, s:s + n] = k.view(b, n, *k.shape[1:])
        self.cache[layer_idx, 1, :, s:s + n] = v.view(b, n, *v.shape[1:])
        kc, vc = self.cache[layer_idx, 0, :, :s + n], self.cache[layer_idx, 1, :, :s + n]
        key_idx = torch.arange(s + n, device=q.device)
        q_idx = s + torch.arange(n, device=q.device)
        causal = key_idx[None, :] <= q_idx[:, None]                                  # [n, s+n]
        mask = causal[None] & self.key_mask[:, None, :s + n]                         # [B, n, s+n]
        mask |= (key_idx[None, :] == q_idx[:, None])[None]   # padding queries see themselves: no NaN rows
        out = masked_attention(q, kc, vc, mask, self.scale)
        return out.reshape(b * n, *out.shape[2:])


class StaticBatchEngine:
    def __init__(self, cfg: EngineConfig, model=None, tokenizer=None, metrics: Metrics | None = None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device) if model is None else model.embed_tokens.weight.device
        self.dtype = resolve_dtype(cfg.dtype, self.device) if model is None else model.embed_tokens.weight.dtype
        if model is None:
            path = resolve_model_path(cfg.model)
            model = load_model(path, self.dtype, self.device, max_positions=cfg.max_model_len)
            tokenizer = tokenizer or load_tokenizer(path)
        self.model, self.tokenizer, self.metrics = model, tokenizer, metrics
        self.dtype = model.embed_tokens.weight.dtype     # the KV cache always matches the weights
        self.eos_ids = set(model.cfg.eos_token_ids)
        self.scale = model.cfg.head_dim ** -0.5
        self.waiting: deque[Sequence] = deque()
        self.seqs: dict[str, Sequence] = {}
        self.batch: list[Sequence] = []
        self._flush = False                 # offline generate(): do not wait W ms for stragglers
        self._rng = torch.Generator().manual_seed(cfg.seed)
        self.stats = {"batches": 0, "steps": 0, "slot_steps": 0, "useful_slot_steps": 0,
                      "peak_running": 0, "generated_tokens": 0, "busy_time": 0.0}

    # --- same interface as LLMEngine -------------------------------------------------------------

    def add_request(self, request_id: str, prompt_ids: list[int], params: SamplingParams,
                    arrival_time: float | None = None) -> Sequence:
        if not prompt_ids or len(prompt_ids) >= self.cfg.max_model_len:
            raise RequestRejected(f"prompt length {len(prompt_ids)} not in [1, {self.cfg.max_model_len})")
        seq = Sequence(request_id, list(prompt_ids), params,
                       arrival_time=time.monotonic() if arrival_time is None else arrival_time)
        if not params.greedy:
            seq.generator = make_generator(params.seed, self.device, self._rng)
        self.seqs[request_id] = seq
        self.waiting.append(seq)
        return seq

    def abort_request(self, request_id: str, reason: str = "abort") -> RequestOutput | None:
        seq = self.seqs.pop(request_id, None)
        if seq is None:
            return None
        if seq in self.waiting:
            self.waiting.remove(seq)
        self._finish(seq, reason)          # a row in the running batch keeps computing, unseen
        if self.metrics:
            self.metrics.num_requests_waiting.set(len(self.waiting))
        return self._output(seq, [])

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.batch)

    def has_work(self) -> bool:
        return bool(self.batch) or self._batch_ready()

    def idle_timeout(self) -> float | None:
        if not self.waiting:
            return None
        waited = time.monotonic() - self.waiting[0].arrival_time
        return max(0.0, self.cfg.static_batch_wait_ms / 1e3 - waited)

    def _batch_ready(self) -> bool:
        if not self.waiting:
            return False
        if len(self.waiting) >= self.cfg.static_batch_size or self._flush:
            return True
        return time.monotonic() - self.waiting[0].arrival_time >= self.cfg.static_batch_wait_ms / 1e3

    # --- steps --------------------------------------------------------------------------------------

    @torch.inference_mode()
    def step(self) -> list[RequestOutput]:
        outputs: list[RequestOutput] = []
        if self.cfg.request_timeout:
            deadline = time.monotonic() - self.cfg.request_timeout
            for seq in [s for s in self.seqs.values() if s.arrival_time < deadline]:
                outputs.append(self.abort_request(seq.id, "timeout"))
        t0 = time.perf_counter()
        if not self.batch:
            if not self._batch_ready():
                return outputs
            tokens = self._start_batch()
        else:
            tokens = self._decode_step()
        now = time.monotonic()
        active = 0
        for seq, tok in zip(self.batch, tokens):
            if seq.status == Status.FINISHED:
                continue                                   # finished rows keep computing; discard
            active += 1
            seq.output_ids.append(tok)
            if seq.first_token_time is None:
                seq.first_token_time = now
            reason = self._stop_reason(seq, tok)
            if reason:
                self.seqs.pop(seq.id, None)
                self._finish(seq, reason, now)
            outputs.append(self._output(seq, [tok]))
        st = self.stats
        st["steps"] += 1
        st["busy_time"] += time.perf_counter() - t0
        st["generated_tokens"] += active
        if self._step_idx > 0:                             # waste counts decode steps
            st["slot_steps"] += len(self.batch)
            st["useful_slot_steps"] += active
        self._step_idx += 1
        if self.metrics:
            self.metrics.step_duration_seconds.observe(time.perf_counter() - t0)
            self.metrics.batch_size.observe(len(self.batch))
            self.metrics.generation_tokens_total.inc(active)
            self.metrics.num_requests_waiting.set(len(self.waiting))
            self.metrics.num_requests_running.set(sum(s.status != Status.FINISHED for s in self.batch))
        if all(s.status == Status.FINISHED for s in self.batch):
            self.batch = []
            self.cache = None
        return outputs

    def _start_batch(self) -> list[int]:
        n = min(len(self.waiting), self.cfg.static_batch_size)
        self.batch = [self.waiting.popleft() for _ in range(n)]
        for s in self.batch:
            s.status = Status.RUNNING
        self.stats["batches"] += 1
        self.stats["peak_running"] = max(self.stats["peak_running"], n)
        plen = max(s.num_prompt_tokens for s in self.batch)
        max_new = min(max(s.params.max_tokens for s in self.batch), self.cfg.max_model_len - plen)
        self.plen, self._step_idx = plen, 0
        cfg = self.model.cfg
        self.cache = torch.zeros(cfg.num_hidden_layers, 2, n, plen + max_new, cfg.num_key_value_heads,
                                 cfg.head_dim, dtype=self.dtype, device=self.device)
        mask = torch.zeros(n, plen + max_new, dtype=torch.bool)
        ids = torch.zeros(n, plen, dtype=torch.long)
        for b, s in enumerate(self.batch):
            p = s.num_prompt_tokens
            ids[b, plen - p:] = torch.tensor(s.prompt_ids)
            mask[b, plen - p:plen] = True
        positions = (mask[:, :plen].long().cumsum(-1) - 1).clamp(min=0)
        self.key_mask = mask.to(self.device)
        self.next_pos = torch.tensor([s.num_prompt_tokens for s in self.batch], device=self.device)
        if self.metrics:
            self.metrics.prompt_tokens_total.inc(sum(s.num_prompt_tokens for s in self.batch))
        ctx = PaddedContext(self.cache, self.key_mask, n, 0, plen, self.scale)
        hidden = self.model(ids.view(-1).to(self.device), positions.view(-1).to(self.device), ctx)
        last = torch.arange(n, device=self.device) * plen + plen - 1
        return self._sample(self.model.compute_logits(hidden[last]))

    def _decode_step(self) -> list[int]:
        n, col = len(self.batch), self.plen + self._step_idx - 1
        self.key_mask[:, col] = True
        ids = torch.tensor([s.last_token for s in self.batch], device=self.device)
        ctx = PaddedContext(self.cache, self.key_mask, n, col, 1, self.scale)
        hidden = self.model(ids, self.next_pos, ctx)
        self.next_pos = self.next_pos + 1
        return self._sample(self.model.compute_logits(hidden))

    def _sample(self, logits) -> list[int]:
        return sample(logits, [s.params for s in self.batch], [s.generator for s in self.batch])

    def _stop_reason(self, seq: Sequence, tok: int) -> str | None:
        if tok in self.eos_ids and not seq.params.ignore_eos:
            return "stop"
        if seq.num_output_tokens >= seq.params.max_tokens or seq.length >= self.cfg.max_model_len:
            return "length"
        # The batch's cache was sized for the longest max_tokens at batch start.
        if self.batch and self.plen + seq.num_output_tokens >= self.cache.shape[3]:
            return "length"
        return None

    def _finish(self, seq: Sequence, reason: str, now: float | None = None) -> None:
        seq.status = Status.FINISHED
        seq.finish_reason = reason
        seq.finish_time = now or time.monotonic()
        if self.metrics:
            self.metrics.observe_finished(seq)

    def _output(self, seq: Sequence, toks: list[int]) -> RequestOutput:
        return RequestOutput(seq.id, toks, seq.is_finished, seq.finish_reason, seq.num_prompt_tokens,
                             seq.num_output_tokens, seq.arrival_time, seq.first_token_time, seq.finish_time,
                             0, None)

    @property
    def waste(self) -> float:
        total = self.stats["slot_steps"]
        return 1.0 - self.stats["useful_slot_steps"] / total if total else 0.0

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]) -> list[list[int]]:
        plist = params if isinstance(params, list) else [params] * len(prompts)
        seqs = []
        for p, sp in zip(prompts, plist):
            self._counter = getattr(self, "_counter", 0) + 1
            seqs.append(self.add_request(f"static-{self._counter}", p, sp))
        self._flush = True
        try:
            while self.has_unfinished():
                self.step()
        finally:
            self._flush = False
        return [s.output_ids for s in seqs]
