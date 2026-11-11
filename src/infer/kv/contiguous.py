"""Slot-based contiguous KV cache (Part 4): every sequence reserves max_seq_len tokens up front.

Shape [layers, 2, max_num_seqs, max_seq_len, kv_heads, head_dim]. It is a paged pool whose
"block" is a whole slot, so writes reuse KVPool.write with slot_mapping = slot * max_seq_len + pos.
Reads slice each slot up to the longest context in the batch instead of gathering blocks.
"""

from __future__ import annotations

from collections import deque

from infer.config import ModelConfig
from infer.kv.paged_cache import AllocStatus, KVPool


class ContiguousKVPool(KVPool):
    def __init__(self, cfg: ModelConfig, num_slots: int, max_seq_len: int, dtype, device):
        super().__init__(cfg, num_slots, max_seq_len, dtype, device)

    def gather(self, layer: int, tables, max_len: int):
        slots = tables[:, 0]
        return self.kv[layer, 0][slots, :max_len], self.kv[layer, 1][slots, :max_len]


class SlotManager:
    """A new sequence takes a free slot; a finished one returns it. Never needs preemption,
    because each slot already holds the maximum length — that is exactly its waste."""

    kind = "contiguous"
    supports_swap = False
    enable_caching = False

    def __init__(self, pool: ContiguousKVPool):
        self.pool = pool
        self.num_slots = pool.num_blocks
        self.max_seq_len = pool.block_size
        self.block_size = pool.block_size          # one "block" per sequence
        self.num_blocks = self.num_slots
        self.free_slots: deque[int] = deque(range(self.num_slots))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_slots)

    def usage(self) -> float:
        return 1.0 - len(self.free_slots) / self.num_slots

    def slot(self, seq, pos: int) -> int:
        return seq.slot * self.max_seq_len + pos

    def allocation_status(self, seq) -> AllocStatus:
        if seq.length > self.max_seq_len:
            return AllocStatus.NEVER
        return AllocStatus.OK if self.free_slots else AllocStatus.LATER

    def allocate(self, seq) -> None:
        seq.slot = self.free_slots.popleft()
        seq.block_table = [seq.slot]
        seq.num_computed_tokens = 0

    def can_append(self, seq) -> bool:
        return seq.length <= self.max_seq_len

    def append_slot(self, seq) -> None:
        pass

    def free(self, seq) -> None:
        if seq.slot is not None:
            self.free_slots.append(seq.slot)
        seq.slot = None
        seq.block_table = []

    def register_computed(self, seq) -> None:
        pass
