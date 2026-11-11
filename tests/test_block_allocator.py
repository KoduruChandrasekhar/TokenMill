"""Allocator properties under random allocate/release sequences (hypothesis)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from infer.kv.block_allocator import BlockAllocator, OutOfBlocks


def test_basic_allocate_release():
    a = BlockAllocator(4)
    ids = a.allocate(3)
    assert len(set(ids)) == 3 and a.num_free == 1
    with pytest.raises(OutOfBlocks):
        a.allocate(2)
    a.release(ids)
    assert a.num_free == 4
    with pytest.raises(RuntimeError, match="double free"):
        a.release([ids[0]])


@settings(max_examples=200, deadline=None)
@given(num_blocks=st.integers(1, 64), ops=st.lists(st.tuples(st.booleans(), st.integers(0, 10)), max_size=80))
def test_random_ops_never_double_allocate(num_blocks, ops):
    """Random sequences never get the same block, never double-free, and free + used == total."""
    a = BlockAllocator(num_blocks)
    owners: dict[int, list[int]] = {}
    next_id = 0
    for is_alloc, n in ops:
        if is_alloc:
            if n > a.num_free:
                with pytest.raises(OutOfBlocks):
                    a.allocate(n)
                continue
            owners[next_id] = a.allocate(n)
            next_id += 1
        elif owners:
            key = sorted(owners)[n % len(owners)]
            a.release(owners.pop(key))
        held = [b for blocks in owners.values() for b in blocks]
        assert len(held) == len(set(held)), "one block given to two sequences"
        assert a.num_free + len(held) == num_blocks
        a.check_invariants()


class CachingAllocatorMachine(RuleBasedStateMachine):
    """Prefix-caching mode: blocks are shared (refcount > 1), parked as evictable at refcount 0,
    revived by share(), and evicted LRU when the free list runs dry."""

    def __init__(self):
        super().__init__()
        self.evicted: list[int] = []
        self.a = BlockAllocator(16, enable_caching=True, on_evict=self._on_evict)
        self.refs: list[int] = []            # every reference we hold (a block may appear many times)

    def _on_evict(self, b):
        # At eviction the block leaves the cache before it is handed out again. (It may be cached and
        # become evictable again later, after its new owner fills and releases it.)
        assert b not in self.a.evictable and b not in self.a.cached
        assert self.a.refcount[b] == 0
        self.evicted.append(b)

    @rule(n=st.integers(1, 4))
    def allocate(self, n):
        if n <= self.a.num_free:
            ids = self.a.allocate(n)
            assert all(self.a.refcount[b] == 1 for b in ids)
            self.refs += ids

    @precondition(lambda self: self.refs)
    @rule(i=st.integers(0, 100))
    def mark_cached(self, i):
        self.a.mark_cached(self.refs[i % len(self.refs)])

    @precondition(lambda self: self.refs)
    @rule(i=st.integers(0, 100))
    def share_live(self, i):
        b = self.refs[i % len(self.refs)]
        if b in self.a.cached:
            self.a.share(b)
            self.refs.append(b)

    @precondition(lambda self: self.a.evictable)
    @rule(i=st.integers(0, 100))
    def revive(self, i):
        b = list(self.a.evictable)[i % len(self.a.evictable)]
        self.a.share(b)
        assert self.a.refcount[b] == 1
        self.refs.append(b)

    @precondition(lambda self: self.refs)
    @rule(i=st.integers(0, 100))
    def release(self, i):
        b = self.refs.pop(i % len(self.refs))
        self.a.release([b])

    @invariant()
    def consistent(self):
        self.a.check_invariants()
        for b in set(self.refs):
            assert self.a.refcount[b] == self.refs.count(b)


TestCachingAllocator = CachingAllocatorMachine.TestCase
TestCachingAllocator.settings = settings(max_examples=100, stateful_step_count=40, deadline=None)
