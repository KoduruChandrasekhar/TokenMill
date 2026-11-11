# Interview notes: the sixteen questions, and where this repo's evidence lives

Answer each one from your own measurements. Numbers are from the Kaggle T4 runs (fp16, three runs
per point) in `bench/results/`; the code pointers show where the mechanism lives.

1. **Why is decode memory-bound and prefill compute-bound?** Decode does ~2 FLOPs per weight byte
   read (one token per pass); prefill reuses each weight across hundreds of tokens. Evidence: batch-1
   decode is 27.3 ms/token eager, 10.2 ms with CUDA graphs (`batch1.json`), against a ~3 ms floor from
   1 GB / 320 GB/s — the rest is overhead, not math.
2. **Why does batching raise decode throughput almost for free?** One weight read per step serves the
   whole batch: static batching goes from 35 tok/s at batch 1 to 504 at batch 16 (`static_batching.json`,
   figure 8) — 14× for 16×.
3. **What does static batching waste?** Finished rows and padding keep computing until the longest row
   ends: 47% of slot-steps on the mixed workload (`waste` in `static_batching.json`). Same 64 requests,
   continuous batching: 1.8× faster offline; served all at once, 3.7× (fixed) and 8× (mixed).
4. **KV cache per token?** 12 KiB = 2 × 24 layers × 2 KV heads × 64 dims × 2 bytes (`ModelConfig.kv_bytes_per_token`).
5. **How does GQA help serving?** 14 query heads share 2 KV heads: 7× smaller cache, 7× less KV read
   per decode step. `decode_attention` exploits it without copying K/V.
6. **Why paged allocation?** Contiguous slots reserve max_model_len; blocks waste at most one partial
   block per sequence. At 256 MiB: 170 concurrent ShareGPT sequences paged vs 10 contiguous (17×),
   2% of allocated KV wasted vs 83% (figure 4). With 12 GB of KV both layouts serve the same (1,701 vs
   1,686 tok/s): paging matters when memory is the constraint.
7. **How did you choose the block size?** Figure 5: 8/16/32 → 0.9/1.9/3.8% waste, same throughput
   within 1%. Gather-based attention makes step time insensitive to block size, so 16.
8. **What happens when KV memory runs out?** `Scheduler._schedule_decodes`: preempt the most recently
   admitted sequence, release (or swap) its blocks, put it at the queue front, recompute prompt +
   generated tokens on resume. 64 MiB pool, ~630 preemptions: recompute 502 vs swap 508 tok/s — a wash
   for a 0.5B model, so recompute is the default (no pinned memory). Preemption shows up as p99 TPOT
   ~6× p50 in the overload runs.
9. **Why can one long prompt hurt everyone's latency?** Its prefill occupies a whole step. Fix: chunked
   prefill (Sarathi-Serve), `--enable-chunked-prefill`: decodes first, prompt chunks in the leftover
   budget (`Scheduler._schedule_chunked`). Mixed workload at 16 req/s, 512-token budget: p99 TPOT
   67 → 50 ms, p99 TTFT 10.4 → 6.4 s, +8% throughput; p50 TTFT +12 ms at light load.
10. **How do you know outputs are correct?** fp32 golden (50 × 64 tokens exact in every engine
    configuration, including forced preemption and chunked prefill), batch invariance, fp16 near-tie
    rule with a measured tolerance of 0.109 (`tests/`; all 5 GPU tests pass on the T4).
11. **Why TTFT and TPOT separately? Goodput?** Waiting for the first token vs reading speed are felt
    differently; goodput counts requests/s meeting both (TTFT ≤ 1 s, TPOT ≤ 100 ms). Decode-first vs
    prefill-first is exactly this trade: at 8 req/s mixed, p99 TPOT 40 vs 55 ms, p50 TTFT 550 vs 53 ms.
12. **Why 52–59% of vLLM, and where is the gap?** Same users in flight (8 → 128), fixed and ShareGPT.
    Without graphs: 23% at one user (27 vs 6 ms per token). Profiler (figure 6): at batch 32 the GPU is
    idle 85 of 95 ms per eager step (~1,250 kernel launches); graphs cut launches to 4 per step. The
    rest: gather-based attention vs a fused paged kernel, per-step metadata building in Python.
13. **Where does the time go in one decode step?** `profile/decode_breakdown_b32*.json`: eager 95 ms =
    85 idle + 5.9 matmul + 1.8 norm/rope/act + 1.2 KV gather + 1.1 attention; graphs 34 ms = 20 idle
    (sampling, metadata between steps) + the same GPU work.
14. **How would prefix caching work?** `BlockManager._block_hashes`: hash(prev hash, 16 token ids) per
    full block, ref-counted sharing, LRU eviction at refcount 0. Shared 1,000-token prefix, all at
    once: 2.5× throughput, p50 TTFT 1.6 s vs 6.9 s.
15. **Client disconnects mid-stream?** Starlette cancels the SSE generator; its `finally` sends an abort
    through the inbox; the engine frees the blocks next step (`test_disconnect_frees_blocks`).
16. **70B to thousands of users?** Tensor parallelism inside a node, pipeline parallelism across nodes
    if needed, many replicas behind a load balancer, prefix-aware routing so shared prefixes hit the
    same replica's cache.

## Stories to prepare

- **A bug hunted (that was not a bug):** the fp32 golden test failed on one prompt in every engine
  configuration — prompt 12, token 11 — while the single-sequence reference path passed. Narrowing it
  down: HF's own full forward pass picks the same token as the engine; only HF's incremental
  `generate()` (which produced the golden file) picks the other. The top-2 gap there is 9.5e-6, far
  below the ~1e-4 fp32 noise between two correct implementations. Fix: `make_golden.py` now records
  every position's fp32 top-2 gap and the test allows divergence only at a recorded tie.
- **A real bug only the GPU showed:** the loader built the model on `meta` in fp32 and copied fp16
  weights into fp32 parameters, so "fp16" ran in fp32; a 30-minute Kaggle probe caught it before the
  long run (`test_half_precision_end_to_end` guards it now).
- **Bugs in the measuring:** a load-generator duration bug (never-sent requests had send time 0) and a
  soak harness that cancelled its own in-flight requests — both found because the numbers disagreed
  with the server's own metrics.
- **An optimization that backfired, three times measured:** graphs at every batch size made 128 users
  16% slower than eager (padded attention in a bandwidth-bound step); finer buckets helped on fixed
  lengths but not ShareGPT and needed 1.3 GB — the first try ran out of memory and silently fell back
  to eager (caught because "graph" numbers matched eager within 1%). Final: graphs only up to batch
  32 — 2.4× eager at 8 users, 1.05–1.18× at 128.
- **A tradeoff measured:** prefill-first vs decode-first, or chunked prefill (numbers above).
