# What I learned building a mini vLLM

*Every number below was measured on one Kaggle T4 (16 GB), fp16, Qwen2.5-0.5B-Instruct, three runs per
point; the raw JSON and the script that turns it into tables are in `bench/results/`.*

I built an LLM inference server from scratch in PyTorch: my own Qwen2.5-0.5B forward pass, continuous
batching, a paged KV cache with preemption, an OpenAI-compatible streaming API, and a benchmark suite
that pits it against naive HuggingFace serving, static batching and vLLM on one free Kaggle T4.
Here is what surprised me.

## 1. Correctness first, and "exact" needs a definition

The rule was: every speed-up must keep the golden test passing — 50 prompts × 64 greedy tokens that
must match HuggingFace in fp32 token for token. It caught real bugs. It also caught a non-bug: one
prompt diverged at token 11 in every engine configuration, while a single-sequence reference passed.
The top two logits there differ by 9.5e-6. HuggingFace disagrees with *itself* at that position: its
incremental `generate()` picks one token, its own full forward pass picks the other. Two correct fp32
implementations differ by ~1e-4 in the logits, so any gap smaller than that is a coin flip. The fix was
to record the top-2 gap of every golden position and allow a divergence only at a recorded tie.

In fp16 the rule changes again: a different batch size changes the order GPU kernels sum in, so tokens
may flip at near-ties. I measured how far HuggingFace's own fp16 moves between batch 1 and batch 8
(max logit difference 0.109 on the T4) and used that as the tolerance.

## 2. A bug only the GPU could show

On CPU everything ran in fp32 and every test passed. On the T4 the static-batching server crashed with
"query is float, key is half". The loader built the model on the `meta` device — in the default dtype,
fp32 — and copied the fp16 checkpoint into fp32 parameters. So `--dtype float16` had silently been
running in fp32, and the "fp16" GPU tests had passed in fp32. A 30-minute "probe" run on Kaggle before
the 12-hour benchmark caught it; a CPU test that loads in half precision now guards it.

## 3. Decode is memory-bound, so batching is (almost) free

Every decode step reads all ~1 GB of fp16 weights; at ~320 GB/s that is ≥ 3 ms whatever the batch.
Static batching shows it directly: 35 tok/s at batch 1, 504 at batch 16 — 14× the throughput for 16×
the batch. But static batching wastes the batch: on a mixed workload (half the requests want 32 tokens,
half 512) 47% of slot-steps went to sequences that had already finished. Given the same 64 requests,
continuous batching finished them 1.8× faster (529 vs 298 tok/s). Served over HTTP with everything
arriving at once, it reached 3.7× static batching's throughput on 512-in/128-out requests (1,686 vs
451 tok/s) and 8× on the mixed workload (2,134 vs 267 tok/s).

## 4. Paged KV cache: the concurrency headline, and when it doesn't matter

KV for this model is 12 KiB per token (2 × 24 layers × 2 KV heads × 64 dims × 2 bytes). Contiguous
slots reserve 2,048 tokens each, so a 256 MiB pool holds 10 sequences. In 16-token blocks the same
pool held 170 ShareGPT sequences at once — 17× — and wasted 2% of what it allocated instead of 83%.
When memory ran out the scheduler preempted the newest sequence and recomputed it later instead of
crashing.

The honest footnote: with the server's real budget (12 GB of KV on the T4) the contiguous cache
already fits 500+ slots, more than the 128-sequence batch cap, and the two layouts served within 1% of
each other (1,701 vs 1,686 tok/s). Paging pays off when memory is the constraint — long contexts,
bigger models, smaller GPUs — not on a 0.5B model with a whole T4 to itself.

## 5. The surprise: Python is the bottleneck, not the GPU

The profiler on a batch-32 decode step: 5.9 ms of matmuls inside a 95 ms step; for 85 ms the GPU sat
idle waiting for Python to launch the next of ~1,250 small kernels. CUDA graphs — record the decode
forward once per (batch, context) bucket, replay it with one launch (4 launch calls per step instead
of 1,206) — cut that step to 34 ms and
batch-1 decode from 27.3 to 10.2 ms per token (HuggingFace `generate()`: 29.1 ms).

Then the graphs made things *worse* under heavy load. With 128 users in flight the graph server
produced 1,362 tok/s against 1,626 without graphs. The context buckets were powers of two, so a batch
whose longest sequence had 640 tokens attended over 1,024 positions — 60% more KV traffic in a
gather-based attention that, at batch 128, is most of the step. Launch overhead stops mattering once
each kernel is big; padding never stops costing.

Buckets every 128 tokens cut the padding and got the fixed workload to 1,541 tok/s at 128 users —
still below eager, and worse than powers of two on ShareGPT. They also need 128 graphs, 1.3 GB of
GPU memory, which the first attempt didn't have: capture ran out of memory next to the KV pool, the
engine fell back to eager as designed, and for one run my "CUDA graphs" numbers were quietly eager
numbers (they matched eager within 1%, which is how I noticed). The lesson from three versions:
graphs are for the launch-bound regime. They now cover decode batches up to 32 and bigger batches run
eagerly, which beats eager at every load measured: 2.4× at 8 users, 1.6× at 32, and 5% at 128
(18% on ShareGPT).

What remains — Python-side step overhead and a gather-based attention instead of a fused
paged-attention kernel — is why my server reaches 52–59% of vLLM's throughput from 8 to 128 users on
the same GPU (without graphs: 23% at one user, 27 ms per token against vLLM's 6).

## 6. Tradeoffs I could measure instead of guess

- **Prefill-first vs decode-first** (mixed workload, 8 req/s): decode-first holds p99 TPOT at 40 ms
  instead of 55, and pays for it in TTFT: p50 550 ms instead of 53.
- **Chunked prefill** (512-token budget): at 16 req/s on the mixed workload p99 TPOT fell from 67 to
  50 ms, p99 TTFT from 10.4 to 6.4 s, and throughput rose 8%; at light load each new request waits
  ~12 ms longer for its first token.
- **Prefix caching** (200 requests sharing a 1,000-token system prompt, all at once): 2.5× the
  throughput (1,735 vs 682 tok/s) and p50 TTFT 1.6 s instead of 6.9 s.
- **Recompute vs swap under overload** (64 MiB pool, ~630 preemptions): 502 vs 508 tok/s — a wash.
  Prefill is cheap for a 0.5B model, so recompute stays the default and nothing needs pinned memory.
- **Block size 8/16/32:** 0.9 / 1.9 / 3.8% waste and the same throughput within 1% — with gather-based
  attention the block size barely moves step time, so 16 stays.

## 7. Two bugs in the measuring, not the server

A 30-minute soak test "failed" with 602 of 6,600 requests lost — about 20 every minute. Server memory
and free KV blocks were identical at every idle point, so I looked at the harness: each round sent 55
seconds of Poisson arrivals and cut the round off at 55 seconds, cancelling whatever was still
streaming. The load generator had a cousin of that bug: when a closed-loop run hit its 15-minute cap,
queued requests that were never sent kept a send time of 0, which stretched the measured duration to
the clock's origin (3 tok/s instead of 37) and marked the run "saturated", skipping the higher
concurrency points. Both are fixed and tested; the rerun soak completed 5,580 of 5,580 requests,
with GPU memory and free KV blocks identical to the byte at all 31 idle checks.

## What I would do next

A Triton paged-attention kernel (read KV blocks in place instead of gathering them), CUDA graphs for
mixed steps, and moving detokenization off the engine's process — the places the profile points at.
