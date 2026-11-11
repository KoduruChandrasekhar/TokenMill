"""Scheduler decisions with fake sequences and no model: admission, budgets, preemption, policies."""

from __future__ import annotations

import torch

from infer.config import EngineConfig, ModelConfig
from infer.engine.scheduler import Scheduler
from infer.engine.sequence import SamplingParams, Sequence, Status
from infer.kv.contiguous import ContiguousKVPool, SlotManager
from infer.kv.paged_cache import BlockManager, KVPool

TINY = ModelConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                   num_attention_heads=2, num_key_value_heads=1)


def make(num_blocks=16, block_size=4, watermark=0.0, swap_blocks=0, caching=False, **cfg_kw):
    cfg = EngineConfig(block_size=block_size, watermark=watermark, **cfg_kw)
    pool = KVPool(TINY, num_blocks, block_size, torch.float32, "cpu")
    swap = KVPool(TINY, swap_blocks, block_size, torch.float32, "cpu") if swap_blocks else None
    cache = BlockManager(pool, watermark, caching, swap)
    return Scheduler(cfg, cache), cache


_n = 0


def seq(prompt_len, max_tokens=100, arrival=0.0) -> Sequence:
    global _n
    _n += 1
    return Sequence(f"s{_n}", list(range(1, prompt_len + 1)), SamplingParams(max_tokens=max_tokens),
                    arrival_time=arrival)


def run_step(sched, plan, token=7):
    """What the engine does after a forward pass: mark tokens computed, append one token each."""
    for s in plan.seqs:
        s.num_computed_tokens = s.length
        s.output_ids.append(token)


def test_prefill_then_decode():
    sched, cache = make()
    a, b = seq(5), seq(3)
    sched.add(a), sched.add(b)
    plan = sched.schedule()
    assert plan.prefill_seqs == [a, b] and not plan.decode_seqs
    assert plan.num_batched_tokens == 8
    assert a.block_table and len(a.block_table) == 2         # ceil(5 / 4)
    run_step(sched, plan)
    plan = sched.schedule()
    assert plan.decode_seqs == [a, b] and not plan.prefill_seqs


def test_admission_respects_watermark():
    sched, cache = make(num_blocks=10, watermark=0.2)          # 2 blocks held back
    a, b = seq(4 * 7), seq(4)                                   # a needs 7 blocks, b needs 1
    sched.add(a), sched.add(b)
    plan = sched.schedule()
    assert plan.prefill_seqs == [a, b]                          # 10 - 7 = 3 >= 2 ok; then 3 - 1 = 2 >= 2 ok
    sched2, _ = make(num_blocks=10, watermark=0.2)
    c, d = seq(4 * 7), seq(4 * 2)
    sched2.add(c), sched2.add(d)
    plan = sched2.schedule()
    assert plan.prefill_seqs == [c]                             # d would leave 1 < 2 free
    assert list(sched2.waiting) == [d]


def test_admission_is_fcfs():
    sched, _ = make(num_blocks=4)
    big, small = seq(4 * 4), seq(2)
    sched.add(big), sched.add(small)
    plan = sched.schedule()
    assert plan.prefill_seqs == [big]                           # small waits behind big (no skipping)


def test_token_budget_holds():
    sched, _ = make(num_blocks=64, max_num_batched_tokens=10)
    a, b, c = seq(4), seq(5), seq(3)
    for s in (a, b, c):
        sched.add(s)
    plan = sched.schedule()
    assert plan.prefill_seqs == [a, b] and plan.num_batched_tokens == 9
    assert list(sched.waiting) == [c]


def test_prompt_bigger_than_budget_runs_alone():
    sched, _ = make(num_blocks=64, max_num_batched_tokens=10)
    a, b = seq(30), seq(2)
    sched.add(a), sched.add(b)
    plan = sched.schedule()
    assert plan.prefill_seqs == [a]


def test_max_num_seqs():
    sched, _ = make(num_blocks=64, max_num_seqs=2)
    for _ in range(4):
        sched.add(seq(2))
    assert len(sched.schedule().prefill_seqs) == 2


def test_never_fits_is_ignored():
    sched, _ = make(num_blocks=4)
    huge = seq(4 * 5)
    sched.add(huge)
    plan = sched.schedule()
    assert plan.ignored == [huge] and plan.empty


def test_growth_allocates_one_block_at_boundary():
    sched, cache = make(num_blocks=16)
    a = seq(4)
    sched.add(a)
    run_step(sched, sched.schedule())                           # length 5 now: needs a 2nd block
    assert len(a.block_table) == 1
    plan = sched.schedule()
    assert plan.decode_seqs == [a] and len(a.block_table) == 2
    run_step(sched, plan)                                       # length 6: still 2 blocks
    sched.schedule()
    assert len(a.block_table) == 2


def test_preemption_picks_newest_and_requeues_at_front():
    sched, cache = make(num_blocks=4)
    old, new = seq(4, arrival=1.0), seq(4, arrival=2.0)
    later = seq(1, arrival=3.0)
    sched.add(old), sched.add(new)
    run_step(sched, sched.schedule())                           # both prefilled: 2 blocks used
    sched.add(later)
    plan = sched.schedule()                                     # later prefills (1 block): 3 used
    assert plan.prefill_seqs == [later]
    run_step(sched, plan)
    # Decode: old and new each need a 2nd block, later needs none. Only 1 block is free.
    plan = sched.schedule()
    assert plan.preempted == [later]
    victim = later
    assert victim.admit_order == max(s.admit_order for s in (old, new, later))   # most recently admitted
    assert victim.status == Status.PREEMPTED and not victim.block_table
    assert sched.waiting[0] is victim                            # front of the queue
    assert old in plan.decode_seqs


def test_preempted_sequence_resumes_with_all_tokens():
    sched, cache = make(num_blocks=3)
    a, b = seq(4), seq(4)
    sched.add(a), sched.add(b)
    run_step(sched, sched.schedule())
    plan = sched.schedule()                                     # both need a 2nd block; 1 free
    assert plan.preempted == [b] and plan.decode_seqs == [a]
    assert b.num_computed_tokens == 0 and b.length == 5
    run_step(sched, plan)
    sched.finish(a, "stop")                                     # a leaves; b can come back
    plan = sched.schedule()
    assert plan.prefill_seqs == [b]
    assert b.num_new_tokens == 5                                # prompt + generated token recomputed
    assert b.arrival_time == 0.0 and b.num_preemptions == 1


def test_finish_frees_blocks():
    sched, cache = make(num_blocks=8)
    a = seq(9)
    sched.add(a)
    sched.schedule()
    assert cache.num_free_blocks == 5
    sched.finish(a, "abort")
    assert cache.num_free_blocks == 8 and a.status == Status.FINISHED and not sched.running


def test_decode_first_defers_prefill():
    sched, _ = make(num_blocks=64, policy="decode_first", max_waiting_steps=3)
    a = seq(4)
    sched.add(a)
    run_step(sched, sched.schedule())                           # nothing running: prefill allowed
    b = seq(4)
    sched.add(b)
    for _ in range(3):
        plan = sched.schedule()
        assert plan.decode_seqs == [a] and not plan.prefill_seqs
        run_step(sched, plan)
    plan = sched.schedule()                                      # waited max_waiting_steps decode steps
    assert plan.prefill_seqs == [b]


def test_prefill_first_admits_immediately():
    sched, _ = make(num_blocks=64, policy="prefill_first")
    a = seq(4)
    sched.add(a)
    run_step(sched, sched.schedule())
    b = seq(4)
    sched.add(b)
    assert sched.schedule().prefill_seqs == [b]


def test_swap_preemption_roundtrip():
    sched, cache = make(num_blocks=3, swap_blocks=8, preemption_mode="swap")
    a, b = seq(4), seq(4)
    sched.add(a), sched.add(b)
    plan = sched.schedule()
    cache.pool.kv[:, :, b.block_table[0]] = 3.0                 # pretend b's K/V was written
    run_step(sched, plan)
    plan = sched.schedule()
    assert plan.preempted == [b] and b.status == Status.SWAPPED
    assert b.cpu_block_table and b.num_computed_tokens == 4     # its cache survives on the CPU
    run_step(sched, plan)
    sched.finish(a, "stop")
    plan = sched.schedule()                                      # swapped in, then decodes
    assert plan.decode_seqs == [b] and b.status == Status.RUNNING
    assert torch.all(cache.pool.kv[:, :, b.block_table[0]] == 3.0)
    assert cache.cpu_allocator.num_free == 8


def test_prefix_caching_shares_blocks():
    sched, cache = make(num_blocks=16, caching=True)
    a = seq(9)
    sched.add(a)
    plan = sched.schedule()
    run_step(sched, plan)
    cache.register_computed(a)                                  # 2 full blocks published
    b = Sequence("b", list(range(1, 10)), SamplingParams())     # same 9-token prompt
    sched.add(b)
    plan = sched.schedule()
    assert plan.prefill_seqs == [b]
    assert b.block_table[:2] == a.block_table[:2]               # shared prefix blocks
    assert b.num_computed_tokens == 8 and b.num_new_tokens == 1
    assert cache.allocator.refcount[a.block_table[0]] == 2


def test_contiguous_slots():
    cfg = EngineConfig(max_num_seqs=2)
    pool = ContiguousKVPool(TINY, 2, 32, torch.float32, "cpu")
    sched = Scheduler(cfg, SlotManager(pool))
    a, b, c = seq(5), seq(5), seq(5)
    for s in (a, b, c):
        sched.add(s)
    plan = sched.schedule()
    assert plan.prefill_seqs == [a, b] and {a.slot, b.slot} == {0, 1}
    freed = a.slot
    sched.finish(a, "stop")
    assert sched.schedule().prefill_seqs == [c] and c.slot == freed


def run_chunked_step(sched, plan, token=7):
    """Engine bookkeeping for a mixed step: chunks advance, finished prompts and decodes sample."""
    for s in plan.prefill_seqs:
        s.num_computed_tokens += plan.chunk(s)
    for s in plan.decode_seqs:
        s.num_computed_tokens = s.length
    for s in plan.seqs:
        if s.num_computed_tokens == s.length:
            s.output_ids.append(token)


def test_chunked_prefill_splits_long_prompt_and_keeps_decoding():
    sched, _ = make(num_blocks=64, enable_chunked_prefill=True, max_num_batched_tokens=8)
    short = seq(3)
    sched.add(short)
    run_chunked_step(sched, sched.schedule())                 # short prompt prefilled whole
    long = seq(20)
    sched.add(long)
    chunks = []
    for _ in range(3):
        plan = sched.schedule()
        assert short in plan.decode_seqs                       # the running stream never stalls
        assert plan.num_batched_tokens <= 8
        if long in plan.prefill_seqs:
            chunks.append(plan.chunk(long))
        run_chunked_step(sched, plan)
    assert chunks == [7, 7, 6]                                 # 8-token budget minus 1 decode each step
    assert long.num_output_tokens == 1 and long.num_computed_tokens == 20


def test_chunked_prefill_budget_shared_fcfs():
    sched, _ = make(num_blocks=64, enable_chunked_prefill=True, max_num_batched_tokens=10)
    a, b, c = seq(6), seq(6), seq(6)
    for s in (a, b, c):
        sched.add(s)
    plan = sched.schedule()
    assert [plan.chunk(s) for s in plan.prefill_seqs] == [6, 4]   # a whole, b partly; c waits
    assert list(sched.waiting) == [c]
    run_chunked_step(sched, plan)
    plan = sched.schedule()                                       # a decodes, b finishes, c starts
    assert plan.decode_seqs == [a] and plan.prefill_seqs[:1] == [b] and plan.chunk(b) == 2
    assert plan.prefill_seqs[1:] == [c] and plan.chunk(c) == 6
