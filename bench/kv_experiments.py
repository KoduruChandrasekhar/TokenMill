"""Part 5 experiments (offline, engine only, chat-like lengths, everything submitted at once):

1. concurrency:   KV budget 128..1024 MiB, contiguous vs paged -> max concurrent sequences, throughput
2. fragmentation: wasted cache = (allocated - used) / allocated, averaged over steps
3. block size:    8, 16, 32 -> waste, mean step time, throughput
4. overload:      a small pool that forces preemption, recompute vs swap -> throughput, preemptions

    python -m bench.kv_experiments                # -> bench/results/kv_experiments.json
"""

from __future__ import annotations

import argparse

import numpy as np

from bench.common import chat_like_lengths, engine_cfg, free_engine, load, random_prompts, run_offline, save
from infer.engine.engine import LLMEngine


def _measure(model, tok, n_prompts, lengths, **cfg_kw) -> dict:
    eng = LLMEngine(engine_cfg(**cfg_kw), model=model, tokenizer=tok)
    res = run_offline(eng, n_prompts, [o for _, o in lengths], record_steps=True)
    samples = res.pop("step_samples")
    alloc = np.array([s["allocated_tokens"] for s in samples], dtype=float)
    used = np.array([s["used_tokens"] for s in samples], dtype=float)
    res["fragmentation"] = float(np.mean(1 - used[alloc > 0] / alloc[alloc > 0])) if (alloc > 0).any() else 0.0
    decode_ms = [s["ms"] for s in samples if s["kind"] == "decode"]
    res["mean_step_ms"] = float(np.mean([s["ms"] for s in samples]))
    res["mean_decode_step_ms"] = float(np.mean(decode_ms)) if decode_ms else None
    res["num_blocks"] = eng.cache.num_blocks
    res["kv_mib"] = eng.cache.pool.nbytes / 2**20
    free_engine(eng)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--num-requests", type=int, default=400)
    ap.add_argument("--budgets", default="128,256,512,1024")
    ap.add_argument("--block-sizes", default="8,16,32")
    ap.add_argument("--block-size-budget", type=float, default=512)
    ap.add_argument("--overload-budget", type=float, default=64)
    ap.add_argument("--sharegpt-path", default=None)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()
    model, tok, dev, dt = load(args.model)
    lengths = chat_like_lengths(args.num_requests, tokenizer=tok, sharegpt_path=args.sharegpt_path,
                                max_len=args.max_model_len)
    prompts = random_prompts([p for p, _ in lengths])
    print(f"{len(lengths)} requests, mean prompt {np.mean([p for p, _ in lengths]):.0f}, "
          f"mean output {np.mean([o for _, o in lengths]):.0f} tokens")
    common = dict(max_model_len=args.max_model_len, max_num_seqs=512)

    concurrency = []
    for budget in [float(b) for b in args.budgets.split(",")]:
        for cache in ("contiguous", "paged"):
            r = _measure(model, tok, prompts, lengths, kv_cache_mb=budget, cache_type=cache, **common)
            concurrency.append({"kv_cache_mb": budget, "cache_type": cache, **r})
            print(f"{cache:10s} {budget:6.0f} MiB: peak {r['peak_running']:4d} seqs, "
                  f"{r['throughput_tok_s']:.0f} tok/s, fragmentation {100 * r['fragmentation']:.1f}%")

    block_sizes = []
    for bs in [int(b) for b in args.block_sizes.split(",")]:
        r = _measure(model, tok, prompts, lengths, kv_cache_mb=args.block_size_budget, block_size=bs, **common)
        block_sizes.append({"block_size": bs, **r})
        print(f"block {bs:3d}: waste {100 * r['fragmentation']:.2f}%, step {r['mean_step_ms']:.1f} ms, "
              f"{r['throughput_tok_s']:.0f} tok/s")

    overload = []
    for mode in ("recompute", "swap"):
        r = _measure(model, tok, prompts, lengths, kv_cache_mb=args.overload_budget, preemption_mode=mode,
                     swap_space_mb=2048, **common)
        overload.append({"preemption_mode": mode, **r})
        print(f"{mode:9s} @ {args.overload_budget} MiB: {r['preemptions']} preemptions, "
              f"{r['throughput_tok_s']:.0f} tok/s")

    save("kv_experiments.json", {"concurrency": concurrency, "block_sizes": block_sizes, "overload": overload,
                                 "lengths": lengths, "args": vars(args)})


if __name__ == "__main__":
    main()
