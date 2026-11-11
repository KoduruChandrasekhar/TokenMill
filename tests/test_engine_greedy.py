"""Greedy outputs must not change with batching, caching layout, preemption or scheduling policy.

Tiny model (always runs): every engine configuration reproduces the single-sequence reference
path token for token. Real model (real_model marker): every configuration reproduces
tests/golden.json — HuggingFace fp32 eager greedy outputs — exactly, all 50 prompts at once.
"""

from __future__ import annotations

import random

import pytest
import torch

from helpers import real_cfg, tiny_cfg
from infer.engine.engine import LLMEngine
from infer.engine.sequence import SamplingParams
from infer.engine.static_engine import StaticBatchEngine
from infer.model.qwen2 import greedy_generate_reference

# ---------------------------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------------------------

N_TOKENS = 24


@pytest.fixture(scope="module")
def tiny_prompts(tiny_tokenizer):
    rng = random.Random(0)
    vocab = len(tiny_tokenizer) - 3                     # skip the special tokens at the end
    lengths = [1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 31, 40, 64, 3, 11]
    return [[rng.randrange(vocab) for _ in range(n)] for n in lengths]


@pytest.fixture(scope="module")
def tiny_reference(tiny_model, tiny_prompts):
    return [greedy_generate_reference(tiny_model, p, N_TOKENS) for p in tiny_prompts]


def _params(n=N_TOKENS):
    return SamplingParams(max_tokens=n, ignore_eos=True)


def test_reference_path_matches_hf(tiny_hf_model, tiny_prompts, tiny_reference):
    for prompt, ref in list(zip(tiny_prompts, tiny_reference))[:4]:
        x = torch.tensor([prompt])
        hf = tiny_hf_model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=N_TOKENS,
                                    min_new_tokens=N_TOKENS, do_sample=False, pad_token_id=0)
        # min_new_tokens masks EOS; compare only up to the first EOS the reference produced
        eos = set(tiny_hf_model.generation_config.eos_token_id or [])
        cut = next((i for i, t in enumerate(ref) if t in eos), len(ref))
        assert hf[0, len(prompt):len(prompt) + cut].tolist() == ref[:cut]


@pytest.mark.parametrize("kw", [
    dict(cache_type="paged"),
    dict(cache_type="paged", block_size=1),
    dict(cache_type="paged", block_size=16),
    dict(cache_type="contiguous"),
    dict(cache_type="paged", policy="decode_first", max_waiting_steps=2),
    dict(cache_type="paged", max_num_batched_tokens=40),            # prompts spread over many steps
    dict(cache_type="paged", enable_prefix_caching=True),
    dict(cache_type="paged", enable_chunked_prefill=True, max_num_batched_tokens=16),
    dict(cache_type="paged", enable_chunked_prefill=True, max_num_batched_tokens=5, block_size=3),
    dict(cache_type="paged", enable_chunked_prefill=True, max_num_batched_tokens=24, enable_prefix_caching=True),
    dict(cache_type="contiguous", enable_chunked_prefill=True, max_num_batched_tokens=20),
], ids=["paged", "bs1", "bs16", "contiguous", "decode_first", "small_budget", "prefix_cache",
        "chunked16", "chunked5_bs3", "chunked+prefix", "chunked_contiguous"])
def test_engine_matches_reference(tiny_model, tiny_tokenizer, tiny_prompts, tiny_reference, kw):
    engine = LLMEngine(tiny_cfg(**kw), model=tiny_model, tokenizer=tiny_tokenizer)
    assert engine.generate(tiny_prompts, _params()) == tiny_reference
    assert not engine.scheduler.running and engine.cache.num_free_blocks == engine.cache.num_blocks


@pytest.mark.parametrize("mode,chunked", [("recompute", False), ("swap", False), ("recompute", True),
                                          ("swap", True)])
def test_forced_preemption_keeps_outputs(tiny_model, tiny_tokenizer, tiny_prompts, tiny_reference, mode, chunked):
    # 24 blocks of 4 tokens: far less than the 16 sequences need at once.
    kv_mb = 24 * 4 * tiny_model.cfg.kv_bytes_per_token(4) / 2**20
    engine = LLMEngine(tiny_cfg(kv_cache_mb=kv_mb, preemption_mode=mode, swap_space_mb=1,
                                enable_chunked_prefill=chunked, max_num_batched_tokens=12 if chunked else 512),
                       model=tiny_model, tokenizer=tiny_tokenizer)
    assert engine.generate(tiny_prompts, _params()) == tiny_reference
    assert engine.scheduler.num_preemptions > 0
    assert engine.cache.num_free_blocks == engine.cache.num_blocks
    if mode == "swap":
        assert engine.cache.cpu_allocator.num_free == engine.cache.cpu_allocator.num_blocks


@pytest.mark.parametrize("step,buckets,max_batch,batch_sizes", [
    (0, [16, 32, 64, 128], 32, [1, 2, 4, 8, 12]),
    (24, [24, 48, 72, 96, 120, 128], 32, [1, 2, 4, 8, 12]),
    (24, [24, 48, 72, 96, 120, 128], 6, [1, 2, 4, 6]),          # batches of 7-12 decode eagerly
])
def test_graph_runner_padding_logic_on_cpu(tiny_model, tiny_tokenizer, tiny_prompts, tiny_reference, monkeypatch,
                                           step, buckets, max_batch, batch_sizes):
    """The CUDA-graph decode path (padded batch sizes, context buckets, scratch block for padded
    rows, static buffers) run eagerly on CPU: same tokens as the reference. Capture itself is
    covered by tests/test_gpu.py. step 0 = power-of-two buckets, else every `step` tokens;
    batches above cuda_graph_max_batch take the ordinary eager path."""
    from infer.engine import cuda_graphs
    from infer.engine.cuda_graphs import CudaGraphRunner

    monkeypatch.setattr(cuda_graphs, "CONTEXT_BUCKETS", [16, 32, 64])   # several buckets below 128
    cfg = tiny_cfg(cuda_graphs=True, max_model_len=128, max_num_seqs=12, cuda_graph_context_step=step,
                   cuda_graph_max_batch=max_batch)
    engine = LLMEngine(cfg, model=tiny_model, tokenizer=tiny_tokenizer)          # graphs stay off on CPU
    assert engine.cache.num_blocks == engine.cache.pool.num_blocks - 1           # scratch block reserved
    runner = CudaGraphRunner(tiny_model, engine.cache, cfg)
    assert runner.batch_sizes == batch_sizes and runner.buckets == buckets
    engine.runner.graphs = runner
    calls = []
    orig = runner.run
    runner.run = lambda seqs: calls.append(len(seqs)) or orig(seqs)
    lens = [N_TOKENS - (i % 6) * 3 for i in range(len(tiny_prompts))]     # finishes shrink the batch
    outs = engine.generate(tiny_prompts, [_params(n) for n in lens])
    assert outs == [r[:n] for r, n in zip(tiny_reference, lens)]
    assert calls and any(n not in runner.batch_sizes for n in calls)              # padding exercised
    assert max(calls) <= batch_sizes[-1]                         # bigger batches went the eager way


def test_half_precision_end_to_end(tiny_model_dir, tiny_tokenizer, tiny_prompts):
    """Weights really take the requested dtype, the KV cache matches them, and both engines run
    when they load the model themselves (the server path). Regression: the loader once copied
    fp16 weights into fp32 parameters, so "fp16" ran in fp32 and broke against an fp16 cache."""
    from infer.model.loader import load_model

    m16 = load_model(tiny_model_dir, torch.float16, torch.device("cpu"))
    assert {p.dtype for p in m16.parameters()} == {torch.float16}
    assert m16.rope_cos.dtype == torch.float32                    # cast at use, as HF does
    for cls in (LLMEngine, StaticBatchEngine):
        eng = cls(tiny_cfg(model=str(tiny_model_dir), dtype="bfloat16"))
        assert eng.model.embed_tokens.weight.dtype == torch.bfloat16 and eng.dtype == torch.bfloat16
        if cls is LLMEngine:
            assert eng.cache.pool.kv.dtype == torch.bfloat16
        outs = eng.generate(tiny_prompts[:5], _params(6))
        assert [len(o) for o in outs] == [6] * 5


def test_static_batching_matches_reference(tiny_model, tiny_tokenizer, tiny_prompts, tiny_reference):
    engine = StaticBatchEngine(tiny_cfg(static_batch_size=6), model=tiny_model, tokenizer=tiny_tokenizer)
    lens = [N_TOKENS - (i % 5) * 4 for i in range(len(tiny_prompts))]     # mixed lengths -> waste
    outs = engine.generate(tiny_prompts, [_params(n) for n in lens])
    assert outs == [r[:n] for r, n in zip(tiny_reference, lens)]
    assert engine.stats["batches"] == 3 and 0 < engine.waste < 1


def test_staggered_arrivals(tiny_model, tiny_tokenizer, tiny_prompts, tiny_reference):
    """New requests join a running batch at every step and still get the same tokens."""
    engine = LLMEngine(tiny_cfg(), model=tiny_model, tokenizer=tiny_tokenizer)
    seqs = []
    pending = list(enumerate(tiny_prompts))
    while pending or engine.has_unfinished():
        if pending:
            i, p = pending.pop(0)
            seqs.append((i, engine.add_request(f"r{i}", p, _params())))
        engine.step()
    for i, s in seqs:
        assert s.output_ids == tiny_reference[i]


def test_shared_prefix_hits_cache(tiny_model, tiny_tokenizer):
    engine = LLMEngine(tiny_cfg(enable_prefix_caching=True), model=tiny_model, tokenizer=tiny_tokenizer)
    prefix = list(range(10, 50))
    prompts = [prefix + [i, i + 1] for i in range(60, 68)]
    ref = [greedy_generate_reference(tiny_model, p, 8) for p in prompts]
    first = engine.generate(prompts[:1], _params(8))
    rest = engine.generate(prompts[1:], _params(8))
    assert first + rest == ref
    assert engine.cache.prefix_hit_tokens >= 7 * 40 - 7 * 4


def test_abort_frees_blocks(tiny_model, tiny_tokenizer, tiny_prompts):
    engine = LLMEngine(tiny_cfg(), model=tiny_model, tokenizer=tiny_tokenizer)
    for i, p in enumerate(tiny_prompts):
        engine.add_request(f"a{i}", p, _params(200))
    for _ in range(3):
        engine.step()
    assert engine.cache.num_free_blocks < engine.cache.num_blocks
    for i in range(len(tiny_prompts)):
        out = engine.abort_request(f"a{i}")
        assert out.finished and out.finish_reason == "abort"
    assert engine.cache.num_free_blocks == engine.cache.num_blocks and not engine.has_unfinished()


def test_eos_and_max_tokens_stop(tiny_model, tiny_tokenizer, tiny_prompts, tiny_reference):
    engine = LLMEngine(tiny_cfg(), model=tiny_model, tokenizer=tiny_tokenizer)
    outs = engine.generate(tiny_prompts, SamplingParams(max_tokens=N_TOKENS))
    for out, ref in zip(outs, tiny_reference):
        cut = next((i + 1 for i, t in enumerate(ref) if t in engine.eos_ids), len(ref))
        assert out == ref[:cut]


def test_seeded_sampling_reproducible_across_batches(tiny_model, tiny_tokenizer, tiny_prompts):
    p = SamplingParams(max_tokens=16, temperature=0.9, top_p=0.95, seed=42, ignore_eos=True)
    alone = LLMEngine(tiny_cfg(), model=tiny_model, tokenizer=tiny_tokenizer).generate(tiny_prompts[3:4], p)
    mixed = LLMEngine(tiny_cfg(), model=tiny_model, tokenizer=tiny_tokenizer).generate(
        tiny_prompts, [p if i == 3 else SamplingParams(max_tokens=16, temperature=1.0) for i in range(16)])
    assert alone[0] == mixed[3]


def test_max_model_len_finishes_with_length(tiny_model, tiny_tokenizer):
    engine = LLMEngine(tiny_cfg(max_model_len=32), model=tiny_model, tokenizer=tiny_tokenizer)
    seq = engine.add_request("x", list(range(20)), _params(100))
    while engine.has_unfinished():
        engine.step()
    assert seq.finish_reason == "length" and seq.length == 32


# ---------------------------------------------------------------------------------------------
# Real model vs golden.json (fp32, exact)
# ---------------------------------------------------------------------------------------------

def _golden_run(engine, golden):
    """Exact match on every prompt. The one exception is a divergence AT a recorded fp32 near-tie
    (top-2 logit gap < 1e-4, where HF's own generate() and full forward pass disagree): after it the
    continuations differ, so the prompt is checked only up to that token. Any other mismatch fails."""
    cases = golden["cases"]
    outs = engine.generate([c["prompt_ids"] for c in cases],
                           SamplingParams(max_tokens=golden["max_new_tokens"]))
    mismatches, at_ties = [], []
    for i, (got, c) in enumerate(zip(outs, cases)):
        want = c["output_ids"]
        if got == want:
            continue
        first = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        if first in c.get("fp32_near_ties", []):
            at_ties.append((i, first))
        else:
            mismatches.append((i, first, c["top2_gap_fp32"][first] if "top2_gap_fp32" in c else None))
    assert len(at_ties) <= sum(bool(c.get("fp32_near_ties")) for c in cases)
    return mismatches, at_ties


def _report(mismatches):
    return "; ".join(f"prompt {i}: first difference at token {j} (fp32 top-2 gap {gap})"
                     for i, j, gap in mismatches[:5])


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.parametrize("kw", [
    dict(cache_type="paged"),
    dict(cache_type="contiguous", kv_cache_mb=1200, max_num_seqs=50, max_model_len=1024),
    dict(cache_type="paged", enable_prefix_caching=True, policy="decode_first"),
    dict(cache_type="paged", enable_chunked_prefill=True, max_num_batched_tokens=96),
], ids=["paged", "contiguous", "prefix_cache+decode_first", "chunked_prefill"])
def test_golden_exact_fp32(real_model, real_tokenizer, golden, kw):
    engine = LLMEngine(real_cfg(**kw), model=real_model, tokenizer=real_tokenizer)
    mismatches, _ = _golden_run(engine, golden)
    assert not mismatches, _report(mismatches)


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["recompute", "swap"])
def test_golden_with_forced_preemption(real_model, real_tokenizer, golden, mode):
    # 120 blocks of 16 tokens (22.5 MiB): 50 prompts need far more, so sequences get preempted.
    engine = LLMEngine(real_cfg(kv_cache_mb=120 * 16 * 12 * 1024 * 2 / 2**20, preemption_mode=mode,
                                swap_space_mb=64), model=real_model, tokenizer=real_tokenizer)
    mismatches, _ = _golden_run(engine, golden)
    assert engine.scheduler.num_preemptions > 0
    assert not mismatches, _report(mismatches)


@pytest.mark.real_model
@pytest.mark.slow
def test_golden_static_batch_16(real_model, real_tokenizer, golden):
    engine = StaticBatchEngine(real_cfg(static_batch_size=16), model=real_model, tokenizer=real_tokenizer)
    mismatches, _ = _golden_run(engine, golden)
    assert not mismatches, _report(mismatches)


@pytest.mark.real_model
def test_golden_reference_path_first_prompts(real_model, golden):
    """The plain Part 2 prefill/decode loop, no engine, on a few golden prompts (incl. long ones)."""
    for c in golden["cases"][:2] + golden["cases"][30:31]:
        out = greedy_generate_reference(real_model, c["prompt_ids"], golden["max_new_tokens"], {151645, 151643})
        assert out == c["output_ids"]
