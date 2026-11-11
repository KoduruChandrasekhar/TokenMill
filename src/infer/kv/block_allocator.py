"""Free list of KV blocks with reference counts.

Reference counts above 1 happen with prefix caching, when sequences that share a prompt prefix
share its blocks. With caching on, a block whose count drops to 0 but holds a full, hashed block
of tokens is parked in an LRU "evictable" list instead of the free list: a later request with the
same prefix can revive it, and it is evicted only when the free list runs dry.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, enable_caching: bool = False,
                 on_evict: Callable[[int], None] | None = None):
        self.num_blocks = num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.refcount = [0] * num_blocks
        self.enable_caching = enable_caching
        self.evictable: OrderedDict[int, None] = OrderedDict()   # refcount 0, content still cached
        self.cached: set[int] = set()                              # blocks that hold a hashed prefix
        self.on_evict = on_evict

    @property
    def num_free(self) -> int:
        """Blocks that allocate() can hand out right now (free + evictable)."""
        return len(self.free) + len(self.evictable)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    def allocate(self, n: int) -> list[int]:
        if n > self.num_free:
            raise OutOfBlocks(f"need {n} blocks, {self.num_free} free")
        ids = []
        for _ in range(n):
            if self.free:
                b = self.free.popleft()
            else:                                       # evict the least recently used cached block
                b, _ = self.evictable.popitem(last=False)
                self.cached.discard(b)
                if self.on_evict:
                    self.on_evict(b)
            assert self.refcount[b] == 0, f"block {b} handed out twice"
            self.refcount[b] = 1
            ids.append(b)
        return ids

    def release(self, ids: list[int]) -> None:
        for b in ids:
            if self.refcount[b] <= 0:
                raise RuntimeError(f"double free of block {b}")
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                if self.enable_caching and b in self.cached:
                    self.evictable[b] = None            # most recently used goes last
                else:
                    self.free.append(b)

    # --- prefix caching -----------------------------------------------------------------------

    def mark_cached(self, b: int) -> None:
        self.cached.add(b)

    def share(self, b: int) -> None:
        """Take another reference to a cached block (reviving it if it was evictable)."""
        if self.refcount[b] == 0:
            if b not in self.evictable:
                raise RuntimeError(f"block {b} is free, not cached")
            del self.evictable[b]
        self.refcount[b] += 1

    def check_invariants(self) -> None:
        free, evict = set(self.free), set(self.evictable)
        assert len(free) == len(self.free), "duplicate block in free list"
        assert not (free & evict), "block both free and evictable"
        for b in range(self.num_blocks):
            in_pool = b in free or b in evict
            assert (self.refcount[b] == 0) == in_pool, f"block {b}: refcount {self.refcount[b]}, in pool {in_pool}"
        assert self.num_free + sum(1 for r in self.refcount if r > 0) == self.num_blocks
