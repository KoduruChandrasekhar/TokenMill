"""Decides, every step, which sequences prefill, which decode, and who gets preempted.

The scheduler never touches the GPU, so it is tested with fake sequences (tests/test_scheduler.py).

Each step is either a prefill step (newly admitted prompts, or preempted sequences being
recomputed) or a decode step (one token for every running sequence):

* prefill_first: whenever a waiting prompt can be admitted, run a prefill step. Lowest time to
  first token; running streams stutter while prompts prefill.
* decode_first: keep decoding while sequences are running; run a prefill step only when nothing
  is running or the waiting queue has been passed over for max_waiting_steps decode steps
  (the same idea as TGI's max_waiting_tokens). Smoother streams, higher time to first token.

With --enable-chunked-prefill (Sarathi-Serve style) steps are mixed instead: every running
sequence decodes, and the token budget left over goes to prompt chunks, oldest first, so a long
prompt is split across steps and never stalls the running streams for a whole step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from infer.config import EngineConfig
from infer.engine.sequence import Sequence, Status
from infer.kv.paged_cache import AllocStatus


@dataclass
class SchedulerOutput:
    prefill_seqs: list[Sequence] = field(default_factory=list)
    decode_seqs: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    swapped_in: list[Sequence] = field(default_factory=list)
    ignored: list[Sequence] = field(default_factory=list)       # can never fit in the KV pool
    prefill_chunks: dict = field(default_factory=dict)          # chunked prefill: tokens this step
    num_batched_tokens: int = 0

    def chunk(self, seq: Sequence) -> int:
        """Tokens a prefilling sequence computes this step."""
        return self.prefill_chunks.get(seq, seq.num_new_tokens)

    @property
    def seqs(self) -> list[Sequence]:
        return self.prefill_seqs + self.decode_seqs

    @property
    def empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs

    @property
    def num_prefill_tokens(self) -> int:
        return sum(self.chunk(s) for s in self.prefill_seqs)


class Scheduler:
    def __init__(self, cfg: EngineConfig, cache):
        self.cfg = cfg
        self.cache = cache                       # BlockManager or SlotManager
        self.waiting: deque[Sequence] = deque()  # FCFS; preempted sequences go back to the front
        self.running: list[Sequence] = []        # admission order, oldest first
        self.num_preemptions = 0
        self._admit_counter = 0
        self._decode_steps_since_prefill = 0

    # --- queue management -----------------------------------------------------------------------

    def add(self, seq: Sequence) -> None:
        seq.status = Status.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def finish(self, seq: Sequence, reason: str) -> None:
        """Release everything the sequence holds. Safe to call for a sequence in any queue."""
        if seq.status == Status.FINISHED:
            return
        if seq in self.running:
            self.running.remove(seq)
            self.cache.free(seq)
        else:
            try:
                self.waiting.remove(seq)
            except ValueError:
                pass
            if seq.status == Status.SWAPPED and seq.cpu_block_table:
                self.cache.cpu_allocator.release(seq.cpu_block_table)
                seq.cpu_block_table = []
        seq.status = Status.FINISHED
        seq.finish_reason = reason

    # --- the plan for one step --------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        if self.cfg.enable_chunked_prefill:
            self._schedule_chunked(out)
            out.num_batched_tokens = len(out.decode_seqs) + out.num_prefill_tokens
            return out
        if self._should_prefill():
            self._schedule_prefills(out)
            if out.prefill_seqs:
                self._decode_steps_since_prefill = 0
                out.num_batched_tokens = out.num_prefill_tokens
                return out
        self._schedule_decodes(out)
        if self.waiting:
            self._decode_steps_since_prefill += 1
        out.num_batched_tokens = len(out.decode_seqs)
        return out

    def _should_prefill(self) -> bool:
        if not self.waiting:
            return False
        if self.cfg.policy == "prefill_first" or not self.running:
            return True
        return self._decode_steps_since_prefill >= self.cfg.max_waiting_steps

    def _admit(self, seq: Sequence) -> None:
        seq.status = Status.RUNNING
        seq.admit_order = self._admit_counter
        self._admit_counter += 1
        self.running.append(seq)

    def _schedule_prefills(self, out: SchedulerOutput) -> None:
        budget = self.cfg.max_num_batched_tokens
        while self.waiting:
            seq = self.waiting[0]
            if len(self.running) >= self.cfg.max_num_seqs:
                break
            if seq.status == Status.SWAPPED:              # resume by copying its blocks back
                if not self.cache.can_swap_in(seq):
                    break
                self.waiting.popleft()
                self.cache.swap_in(seq)
                self._admit(seq)
                out.swapped_in.append(seq)
                continue
            status = self.cache.allocation_status(seq)
            if status == AllocStatus.NEVER:
                self.waiting.popleft()
                out.ignored.append(seq)
                continue
            if status == AllocStatus.LATER:
                break
            # Token budget: a prompt that does not fit waits for the next step, except that a
            # prompt bigger than the whole budget runs alone.
            if seq.length > budget and out.prefill_seqs:
                break
            self.waiting.popleft()
            self.cache.allocate(seq)
            self._admit(seq)
            out.prefill_seqs.append(seq)
            budget -= seq.num_new_tokens
            if budget <= 0:
                break

    def _schedule_decodes(self, out: SchedulerOutput) -> None:
        """Every running sequence decodes one token. A sequence crossing a block boundary needs a
        new block; if none is free, preempt the most recently admitted sequence (LIFO protects
        older requests) until it fits."""
        queue = sorted(self.running, key=lambda s: s.admit_order)
        while queue:
            seq = queue.pop(0)
            while not self.cache.can_append(seq):
                victim = queue.pop() if queue else seq
                self._preempt(victim, out)
                if victim is seq:
                    break
            else:
                self.cache.append_slot(seq)
                out.decode_seqs.append(seq)

    def _schedule_chunked(self, out: SchedulerOutput) -> None:
        budget = self.cfg.max_num_batched_tokens
        # 1. Running sequences, oldest first. Fully prefilled ones decode (growing or preempting as in
        #    _schedule_decodes); ones still part-way through their prompt continue in step 2.
        queue = sorted(self.running, key=lambda s: s.admit_order)
        partial: list[Sequence] = []
        while queue:
            seq = queue.pop(0)
            if seq.num_new_tokens > 1:
                partial.append(seq)
                continue
            while not self.cache.can_append(seq):
                victim = queue.pop() if queue else seq
                self._preempt(victim, out)
                if victim is seq:
                    break
            else:
                self.cache.append_slot(seq)
                out.decode_seqs.append(seq)
        budget -= len(out.decode_seqs)
        # 2. Continue prompts already part-way through, oldest first.
        for seq in partial:
            if budget <= 0:
                break
            self._add_chunk(seq, budget, out)
            budget -= out.prefill_chunks[seq]
        # 3. Admit waiting prompts (FCFS) into what is left of the budget.
        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            seq = self.waiting[0]
            if seq.status == Status.SWAPPED:
                if not self.cache.can_swap_in(seq):
                    break
                self.waiting.popleft()
                self.cache.swap_in(seq)
                self._admit(seq)
                out.swapped_in.append(seq)                # decodes (or continues) next step
                continue
            status = self.cache.allocation_status(seq)
            if status == AllocStatus.NEVER:
                self.waiting.popleft()
                out.ignored.append(seq)
                continue
            if status == AllocStatus.LATER:
                break
            self.waiting.popleft()
            self.cache.allocate(seq)                      # blocks for the whole prompt up front
            self._admit(seq)
            self._add_chunk(seq, budget, out)
            budget -= out.prefill_chunks[seq]

    @staticmethod
    def _add_chunk(seq: Sequence, budget: int, out: SchedulerOutput) -> None:
        out.prefill_seqs.append(seq)
        out.prefill_chunks[seq] = min(seq.num_new_tokens, budget)

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.running.remove(seq)
        if self.cfg.preemption_mode == "swap" and self.cache.supports_swap and self.cache.can_swap_out(seq):
            self.cache.swap_out(seq)
            seq.status = Status.SWAPPED
        else:
            self.cache.free(seq)
            seq.num_computed_tokens = 0          # recompute prompt + generated tokens on resume
            seq.status = Status.PREEMPTED
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)             # front of the queue, original arrival time kept
        out.preempted.append(seq)
