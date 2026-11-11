"""Paged KV cache: one pool of fixed-size blocks, block tables per sequence, slot mapping, gather.

Memory layout: one tensor [layers, 2 (K and V), num_blocks, block_size, kv_heads, head_dim].
Logical block i of a sequence holds token positions i*bs .. i*bs + bs-1, and lives in physical
block block_table[i]. Token position p is written to flat slot block_table[p // bs] * bs + p % bs.
"""

from __future__ import annotations

import enum
import math

import torch

from infer.config import ModelConfig
from infer.kv.block_allocator import BlockAllocator


class AllocStatus(enum.Enum):
    OK = 0
    LATER = 1      # not enough free blocks now
    NEVER = 2      # could not fit even in an empty pool


class KVPool:
    """The cache tensor itself, with a batched write and a gather-based read."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, dtype, device):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.kv = torch.zeros(cfg.num_hidden_layers, 2, num_blocks, block_size, self.kv_heads, self.head_dim,
                              dtype=dtype, device=device)

    @property
    def nbytes(self) -> int:
        return self.kv.numel() * self.kv.element_size()

    def write(self, layer: int, slot_mapping: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write every token of the step at once: view the pool as [num_blocks * bs, KVH, D]."""
        self.kv[layer, 0].view(-1, self.kv_heads, self.head_dim).index_copy_(0, slot_mapping, k)
        self.kv[layer, 1].view(-1, self.kv_heads, self.head_dim).index_copy_(0, slot_mapping, v)

    def gather(self, layer: int, tables: torch.Tensor, max_len: int):
        """tables: [B, W] block ids (short tables padded with any valid id). Returns K and V as
        [B, max_len, KVH, D]. This copies the context into a temporary every step; a kernel that
        reads blocks in place (kernels/paged_attention.py) avoids that traffic."""
        b, w = tables.shape
        k = self.kv[layer, 0][tables].view(b, w * self.block_size, self.kv_heads, self.head_dim)
        v = self.kv[layer, 1][tables].view(b, w * self.block_size, self.kv_heads, self.head_dim)
        return k[:, :max_len], v[:, :max_len]


class BlockManager:
    """Hands blocks to sequences: admission checks with a watermark, growth by one block when a
    sequence crosses a block boundary, release on finish/preempt, optional prefix caching and
    optional swapping of a preempted sequence's blocks to CPU memory."""

    kind = "paged"

    def __init__(self, pool: KVPool, watermark: float = 0.01, enable_caching: bool = False,
                 swap_pool: KVPool | None = None, num_usable_blocks: int | None = None):
        self.pool = pool
        self.block_size = pool.block_size
        # Blocks past num_usable_blocks are never handed out (CUDA graphs use one as a scratch sink).
        self.num_blocks = num_usable_blocks or pool.num_blocks
        self.watermark_blocks = int(watermark * self.num_blocks)
        self.enable_caching = enable_caching
        self.allocator = BlockAllocator(self.num_blocks, enable_caching, on_evict=self._on_evict)
        self.hash_to_block: dict[int, int] = {}
        self.block_to_hash: dict[int, int] = {}
        self.swap_pool = swap_pool
        self.cpu_allocator = BlockAllocator(swap_pool.num_blocks) if swap_pool is not None else None
        self.prefix_hit_tokens = 0
        self.prefix_query_tokens = 0

    # --- bookkeeping ----------------------------------------------------------------------------

    def blocks_for(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    def usage(self) -> float:
        return self.allocator.num_used / self.num_blocks

    def slot(self, seq, pos: int) -> int:
        return seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size

    # --- admission / growth / release ------------------------------------------------------------

    def allocation_status(self, seq) -> AllocStatus:
        need = self.blocks_for(seq.length)
        if need + self.watermark_blocks > self.num_blocks:
            return AllocStatus.NEVER
        need -= self._num_active_prefix_hits(seq)
        if self.allocator.num_free - need >= self.watermark_blocks:
            return AllocStatus.OK
        return AllocStatus.LATER

    def allocate(self, seq) -> None:
        """Give a waiting sequence blocks for all of its tokens (prompt, plus generated tokens
        when it resumes after preemption). Reused prefix blocks count as already computed."""
        assert not seq.block_table, f"{seq.id} still holds blocks"
        table: list[int] = []
        if self.enable_caching:
            hashes = self._block_hashes(seq)
            reusable = (seq.length - 1) // self.block_size        # always compute >= 1 token
            for h in hashes[:reusable]:
                b = self.hash_to_block.get(h)
                if b is None:
                    break
                self.allocator.share(b)
                table.append(b)
            self.prefix_hit_tokens += len(table) * self.block_size
            self.prefix_query_tokens += seq.length
        seq.num_computed_tokens = len(table) * self.block_size
        seq.num_registered_blocks = len(table)
        table += self.allocator.allocate(self.blocks_for(seq.length) - len(table))
        seq.block_table = table

    def can_append(self, seq) -> bool:
        """Room for this step's new token? (A sequence of length L needs ceil(L / bs) blocks.)"""
        return len(seq.block_table) * self.block_size >= seq.length or self.allocator.num_free >= 1

    def append_slot(self, seq) -> None:
        if len(seq.block_table) * self.block_size < seq.length:
            seq.block_table.append(self.allocator.allocate(1)[0])

    def free(self, seq) -> None:
        # Reverse order: with caching, a prefix's tail blocks become evictable before its head.
        self.allocator.release(list(reversed(seq.block_table)))
        seq.block_table = []
        seq.num_registered_blocks = 0

    # --- prefix caching ---------------------------------------------------------------------------

    def _block_hashes(self, seq) -> list[int]:
        """hash(previous block's hash, this block's token ids) for every full block of the sequence."""
        tokens = seq.all_token_ids
        hashes = seq.block_hashes
        for i in range(len(hashes), len(tokens) // self.block_size):
            prev = hashes[-1] if hashes else None
            hashes.append(hash((prev, tuple(tokens[i * self.block_size:(i + 1) * self.block_size]))))
        return hashes

    def _num_active_prefix_hits(self, seq) -> int:
        """Prefix blocks that are in use by someone else, so reusing them costs no free block."""
        if not self.enable_caching:
            return 0
        hits = 0
        for h in self._block_hashes(seq)[:(seq.length - 1) // self.block_size]:
            b = self.hash_to_block.get(h)
            if b is None:
                break
            if self.allocator.refcount[b] > 0:
                hits += 1
        return hits

    def register_computed(self, seq) -> None:
        """After a step: publish every newly filled block so later requests can reuse it."""
        if not self.enable_caching:
            return
        full = seq.num_computed_tokens // self.block_size
        if full <= seq.num_registered_blocks:
            return
        hashes = self._block_hashes(seq)
        for i in range(seq.num_registered_blocks, full):
            h, b = hashes[i], seq.block_table[i]
            if h not in self.hash_to_block and b not in self.block_to_hash:
                self.hash_to_block[h] = b
                self.block_to_hash[b] = h
                self.allocator.mark_cached(b)
        seq.num_registered_blocks = full

    def _on_evict(self, b: int) -> None:
        h = self.block_to_hash.pop(b, None)
        if h is not None:
            self.hash_to_block.pop(h, None)

    # --- swapping ---------------------------------------------------------------------------------

    @property
    def supports_swap(self) -> bool:
        return self.swap_pool is not None

    def can_swap_out(self, seq) -> bool:
        return self.cpu_allocator is not None and self.cpu_allocator.num_free >= len(seq.block_table)

    def swap_out(self, seq) -> None:
        cpu_ids = self.cpu_allocator.allocate(len(seq.block_table))
        src = torch.tensor(seq.block_table, device=self.pool.kv.device)
        dst = torch.tensor(cpu_ids)
        self.swap_pool.kv[:, :, dst] = self.pool.kv[:, :, src].to("cpu", non_blocking=False)
        self.free(seq)
        seq.cpu_block_table = cpu_ids

    def can_swap_in(self, seq) -> bool:
        need = len(seq.cpu_block_table) + 1              # + room for the next token
        return self.allocator.num_free - need >= self.watermark_blocks

    def swap_in(self, seq) -> None:
        gpu_ids = self.allocator.allocate(len(seq.cpu_block_table))
        src = torch.tensor(seq.cpu_block_table)
        dst = torch.tensor(gpu_ids, device=self.pool.kv.device)
        self.pool.kv[:, :, dst] = self.swap_pool.kv[:, :, src].to(self.pool.kv.device, non_blocking=True)
        self.cpu_allocator.release(seq.cpu_block_table)
        seq.cpu_block_table = []
        seq.block_table = gpu_ids
        seq.num_registered_blocks = 0
