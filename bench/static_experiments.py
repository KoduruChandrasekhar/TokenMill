"""Part 3 measurements for static batching (baseline B), offline:

1. Throughput vs batch size (1, 2, 4, 8, 16, 32) for fixed-length requests (512 in, 128 out).
2. Waste on a mixed-length workload (half 32, half 512 output tokens) at batch 16:
   waste = 1 - useful slot-steps / (batch size x decode steps).
   The continuous engine runs the same mixed workload for comparison.

    python -m bench.static_experiments            # -> bench/results/static_batching.json
"""

from __future__ import annotations

import argparse
import random

from bench.common import engine_cfg, free_engine, load, random_prompts, run_offline, save
from infer.engine.engine import LLMEngine
from infer.engine.static_engine import StaticBatchEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    ap.add_argument("--batches-per-size", type=int, default=3)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--mixed-requests", type=int, default=64)
    ap.add_argument("--mixed-batch", type=int, default=16)
    args = ap.parse_args()
    model, tok, dev, dt = load(args.model, args.dtype, args.device)

    curve = []
    for b in [int(x) for x in args.batch_sizes.split(",")]:
        n = b * args.batches_per_size
        eng = StaticBatchEngine(engine_cfg(static_batch_size=b, device=str(dev), dtype=str(dt).split(".")[-1]),
                                model=model, tokenizer=tok)
        eng._flush = True
        res = run_offline(eng, random_prompts([args.input_len] * n, seed=b), [args.output_len] * n)
        curve.append({"batch_size": b, "throughput_tok_s": res["throughput_tok_s"], "wall_s": res["wall_s"]})
        print(f"static batch {b:3d}: {res['throughput_tok_s']:.0f} tok/s")
        free_engine(eng)

    rng = random.Random(0)
    outs = [32] * (args.mixed_requests // 2) + [512] * (args.mixed_requests - args.mixed_requests // 2)
    rng.shuffle(outs)
    prompts = random_prompts([256] * len(outs), seed=1)
    eng = StaticBatchEngine(engine_cfg(static_batch_size=args.mixed_batch, device=str(dev),
                                       dtype=str(dt).split(".")[-1]), model=model, tokenizer=tok)
    eng._flush = True
    static_mixed = run_offline(eng, prompts, outs)
    static_mixed.update(waste=eng.waste, slot_steps=eng.stats["slot_steps"],
                        useful_slot_steps=eng.stats["useful_slot_steps"])
    free_engine(eng)
    print(f"static mixed (batch {args.mixed_batch}): waste {100 * static_mixed['waste']:.1f}%, "
          f"{static_mixed['throughput_tok_s']:.0f} tok/s")
    eng = LLMEngine(engine_cfg(device=str(dev), dtype=str(dt).split(".")[-1], max_num_seqs=args.mixed_batch),
                    model=model, tokenizer=tok)
    cont_mixed = run_offline(eng, prompts, outs)
    free_engine(eng)
    print(f"continuous mixed (max {args.mixed_batch} seqs): {cont_mixed['throughput_tok_s']:.0f} tok/s")
    for r in (static_mixed, cont_mixed):
        r.pop("step_samples", None)
    save("static_batching.json", {"throughput_vs_batch": curve, "mixed_static": static_mixed,
                                  "mixed_continuous": cont_mixed, "args": vars(args)})


if __name__ == "__main__":
    main()
