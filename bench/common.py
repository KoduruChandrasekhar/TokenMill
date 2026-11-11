"""Helpers shared by the offline (no HTTP) experiments."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from infer.config import EngineConfig
from infer.engine.sequence import SamplingParams
from infer.model.loader import load_model, load_tokenizer, resolve_device, resolve_dtype, resolve_model_path

RESULTS = Path(__file__).resolve().parent / "results"


def load(model: str, dtype: str = "auto", device: str = "auto", max_positions: int = 4096):
    dev = resolve_device(device)
    dt = resolve_dtype(dtype, dev)
    path = resolve_model_path(model)
    return load_model(path, dt, dev, max_positions=max_positions), load_tokenizer(path), dev, dt


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def random_prompts(lengths: list[int], vocab: int = 150000, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1000, vocab) for _ in range(n)] for n in lengths]


def chat_like_lengths(n: int, seed: int = 0, sharegpt_path: str | None = None, tokenizer=None,
                      max_len: int = 2048) -> list[tuple[int, int]]:
    """(prompt_len, output_len) pairs: real ShareGPT lengths when available, else a lognormal fit
    with a typical chat sequence near 500 tokens."""
    if sharegpt_path or tokenizer is not None:
        try:
            from bench.datasets import sharegpt

            reqs = sharegpt(tokenizer, n, seed=seed, path=sharegpt_path, max_model_len=max_len)
            return [(r.prompt_len, r.output_len) for r in reqs]
        except Exception as e:  # offline / no file: fall back
            print(f"ShareGPT unavailable ({e}); using synthetic lengths")
    rng = np.random.default_rng(seed)
    p = np.clip(rng.lognormal(5.2, 0.9, n), 8, 1024).astype(int)
    o = np.clip(rng.lognormal(5.3, 0.8, n), 8, 1024).astype(int)
    o = np.minimum(o, max_len - p - 1)
    return list(zip(p.tolist(), o.tolist()))


def run_offline(engine, prompts: list[list[int]], out_lens: list[int], record_steps: bool = False) -> dict:
    """Submit everything at once, step to completion. Returns throughput and per-step samples."""
    for i, (p, o) in enumerate(zip(prompts, out_lens)):
        engine.add_request(f"x{i}", p, SamplingParams(max_tokens=o, ignore_eos=True))
    steps = []
    t0 = time.perf_counter()
    while engine.has_unfinished():
        ts = time.perf_counter()
        n_prefill = engine.stats.get("prefill_steps", 0)
        outs = engine.step()
        if record_steps:
            kind = "prefill" if engine.stats.get("prefill_steps", 0) > n_prefill else "decode"
            steps.append({"t": ts - t0, "ms": 1e3 * (time.perf_counter() - ts), "tokens": len(outs),
                          "kind": kind, **kv_snapshot(engine)})
    sync(engine.device)
    wall = time.perf_counter() - t0
    gen = sum(out_lens)
    return {"wall_s": wall, "generated_tokens": gen, "throughput_tok_s": gen / wall,
            "peak_running": engine.stats["peak_running"], "steps": engine.stats["steps"],
            "preemptions": getattr(getattr(engine, "scheduler", None), "num_preemptions", 0),
            "step_samples": steps}


def kv_snapshot(engine) -> dict:
    """Allocated vs used KV token slots across running sequences (fragmentation)."""
    sched = getattr(engine, "scheduler", None)
    if sched is None:
        return {}
    cache = engine.cache
    used = sum(s.num_computed_tokens for s in sched.running)
    if cache.kind == "contiguous":
        allocated = len(sched.running) * cache.max_seq_len
    else:
        allocated = sum(len(s.block_table) for s in sched.running) * cache.block_size
    return {"running": len(sched.running), "allocated_tokens": allocated, "used_tokens": used,
            "kv_usage": cache.usage()}


def engine_cfg(**kw) -> EngineConfig:
    base = dict(request_timeout=None, max_num_seqs=256, max_num_batched_tokens=4096, max_model_len=2048)
    base.update(kw)
    return EngineConfig(**base)


def save(name: str, doc: dict) -> Path:
    from bench.load_gen import environment

    doc = {**doc, "env": environment()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(doc, indent=1))
    print(f"wrote {path}")
    return path


def free_engine(engine) -> None:
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
