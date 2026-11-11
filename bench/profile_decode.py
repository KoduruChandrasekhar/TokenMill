"""Where does one decode step's time go? Wraps 50 decode steps in torch.profiler.

    python -m bench.profile_decode --batch 32           # -> bench/results/profile/*.json

Writes a Chrome/Perfetto trace (open in chrome://tracing or ui.perfetto.dev) and a summary that
splits each step into kernel categories plus "GPU idle": wall time not covered by any GPU kernel,
which for a model this small is mostly Python and kernel-launch overhead.
For a lower-level view: nsys profile -o decode python -m bench.profile_decode
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from bench.common import RESULTS, engine_cfg, load, random_prompts, sync
from infer.engine.engine import LLMEngine
from infer.engine.sequence import SamplingParams

CATEGORIES = [                              # first match wins (lower-cased op / kernel names)
    ("attention", ("scaled_dot_product", "attention", "fmha", "softmax")),
    ("kv gather", ("index_select", "aten::index", "gather", "indexfunc", "index_elementwise")),
    ("kv write", ("index_copy", "indexcopy", "index_put")),
    ("matmul", ("gemm", "aten::mm", "aten::addmm", "aten::linear", "matmul", "cutlass", "sm75_xmma", "turing")),
    ("sampling", ("argmax", "sort", "multinomial", "topk", "cumsum")),
    ("norm / rope / act", ("rsqrt", "silu", "mul", "add", "cat", "neg", "pow", "mean", "embedding")),
    ("copies", ("copy", "to", "contiguous", "memcpy")),
]


def categorize(name: str) -> str:
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cuda-graphs", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    model, tok, dev, dt = load(args.model)
    eng = LLMEngine(engine_cfg(kv_cache_mb=1024, cuda_graphs=args.cuda_graphs), model=model, tokenizer=tok)
    for i, p in enumerate(random_prompts([args.prompt_len] * args.batch)):
        eng.add_request(f"p{i}", p, SamplingParams(max_tokens=10_000 // 8, ignore_eos=True))
    while eng.stats["decode_steps"] < 10:                   # prefill + warm-up decode steps
        eng.step()
    sync(dev)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    t0 = time.perf_counter()
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(args.steps):
            eng.step()
        sync(dev)
    wall = time.perf_counter() - t0
    out_dir = RESULTS / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    trace = out_dir / f"decode_trace_b{args.batch}{suffix}.json"
    prof.export_chrome_trace(str(trace))

    by_cat: dict[str, float] = defaultdict(float)
    if dev.type == "cuda":
        kernel_total = 0.0
        for evt in prof.events():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                dur = getattr(evt, "device_time_total", None)
                dur = evt.cuda_time_total if dur is None else dur
                by_cat[categorize(evt.name)] += dur / 1e3
                kernel_total += dur / 1e3
        by_cat["GPU idle (Python + launch overhead)"] = max(0.0, wall * 1e3 - kernel_total)
    else:
        for evt in prof.key_averages():
            if evt.key.startswith("aten::") and evt.self_cpu_time_total > 0:
                by_cat[categorize(evt.key)] += evt.self_cpu_time_total / 1e3
        by_cat["python / framework overhead"] = max(0.0, wall * 1e3 - sum(by_cat.values()))
    per_step = {k: v / args.steps for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])}
    summary = {"batch": args.batch, "prompt_len": args.prompt_len, "steps": args.steps, "device": str(dev),
               "cuda_graphs": args.cuda_graphs, "wall_ms_per_step": 1e3 * wall / args.steps,
               "ms_per_step_by_category": per_step, "trace": trace.name}
    (out_dir / f"decode_breakdown_b{args.batch}{suffix}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    keys = ["self_device_time_total", "self_cuda_time_total"] if dev.type == "cuda" else ["self_cpu_time_total"]
    for key in keys:                       # the CUDA sort key was renamed in newer PyTorch
        try:
            print(prof.key_averages().table(sort_by=key, row_limit=15))
            break
        except Exception:
            continue


if __name__ == "__main__":
    main()
