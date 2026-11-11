"""Batch-1 numbers: time to first token, decode ms per token, tokens per second.

Part 1 records them for HuggingFace generate(); Part 2's done criterion is our engine within
about 1.5x of HuggingFace at batch 1.

    python -m bench.batch1                         # -> bench/results/batch1.json
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from bench.common import engine_cfg, load, save, sync
from infer.engine.engine import LLMEngine
from infer.engine.sequence import SamplingParams


def time_hf(model_name: str, dev, dt, prompt_ids: list[int], n_out: int, reps: int) -> dict:
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt).to(dev).eval()
    x = torch.tensor([prompt_ids], device=dev)
    kw = dict(attention_mask=torch.ones_like(x), do_sample=False, temperature=None, top_p=None, top_k=None,
              repetition_penalty=1.0, pad_token_id=0)
    ttfts, totals = [], []
    with torch.inference_mode():
        hf.generate(x, max_new_tokens=8, min_new_tokens=8, **kw)          # warm-up
        for _ in range(reps):
            sync(dev)
            t0 = time.perf_counter()
            hf.generate(x, max_new_tokens=1, min_new_tokens=1, **kw)
            sync(dev)
            ttfts.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            hf.generate(x, max_new_tokens=n_out, min_new_tokens=n_out, **kw)
            sync(dev)
            totals.append(time.perf_counter() - t0)
    del hf
    return _summ(ttfts, totals, n_out)


def time_engine(model, tok, prompt_ids: list[int], n_out: int, reps: int, **cfg) -> dict:
    eng = LLMEngine(engine_cfg(kv_cache_mb=256, **cfg), model=model, tokenizer=tok)
    eng.generate([prompt_ids], SamplingParams(max_tokens=8, ignore_eos=True))       # warm-up
    ttfts, totals = [], []
    for _ in range(reps):
        sync(eng.device)
        t0 = time.perf_counter()
        seq = eng.add_request(f"b1-{time.time_ns()}", prompt_ids, SamplingParams(max_tokens=n_out, ignore_eos=True))
        eng.step()                                                                   # prefill -> 1st token
        ttfts.append(time.perf_counter() - t0)
        while not seq.is_finished:
            eng.step()
        sync(eng.device)
        totals.append(time.perf_counter() - t0)
    return _summ(ttfts, totals, n_out)


def _summ(ttfts, totals, n_out) -> dict:
    ttft, total = float(np.median(ttfts)), float(np.median(totals))
    decode_ms = 1e3 * (total - ttft) / (n_out - 1)
    return {"ttft_ms": 1e3 * ttft, "decode_ms_per_token": decode_ms, "tokens_per_s": n_out / total,
            "decode_tokens_per_s": 1e3 / decode_ms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()
    model, tok, dev, dt = load(args.model)
    prompt = [int(x) for x in torch.randint(1000, 100000, (args.prompt_len,), generator=torch.Generator().manual_seed(0))]
    results = {"ours_paged": time_engine(model, tok, prompt, args.output_len, args.reps)}
    if dev.type == "cuda":
        results["ours_paged_cuda_graphs"] = time_engine(model, tok, prompt, args.output_len, args.reps,
                                                        cuda_graphs=True)
    if not args.skip_hf:
        results["hf_generate"] = time_hf(args.model, dev, dt, prompt, args.output_len, args.reps)
        results["ours_vs_hf_decode_ratio"] = (results["ours_paged"]["decode_ms_per_token"]
                                              / results["hf_generate"]["decode_ms_per_token"])
    for k, v in results.items():
        print(k, v)
    save("batch1.json", {"results": results, "args": vars(args), "device": str(dev), "dtype": str(dt)})


if __name__ == "__main__":
    main()
