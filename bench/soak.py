"""Soak test: random-arrival load for 30 minutes against a running server; no memory growth, no
stuck requests, no errors.

    python -m bench.soak --base-url http://localhost:8000 --minutes 30 --rate 4

The load comes in rounds: 45 s of Poisson arrivals, then every request is allowed to finish, then
the server must go idle and /metrics is sampled (gpu_memory_allocated_bytes, kv_free_blocks).
At every idle point allocated memory and free blocks must return to the first idle point's level,
and no request may fail. (A first version cut each round off at 55 s while still sending 55 s
worth of arrivals, so the last ~10% of requests were cancelled by the harness itself.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter

import httpx
import numpy as np

from bench.common import RESULTS


def _metric(text: str, name: str) -> float | None:
    m = re.search(rf"^{name}(?:\{{[^}}]*\}})? (\S+)", text, re.M)
    return float(m.group(1)) if m else None


async def main_async(args) -> int:
    from transformers import AutoTokenizer

    from bench.datasets import load_workload
    from bench.load_gen import run_load, summarize

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pool = load_workload("mixed", tok, 200, seed=1, input_len=128, short=16, long=256)
    rng = np.random.default_rng(0)
    samples, totals = [], {"completed": 0, "failed": 0}
    errors: Counter = Counter()
    t_end = time.time() + 60 * args.minutes
    minute = 0
    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() < t_end:
            reqs = [pool[i] for i in rng.integers(0, len(pool), int(args.rate * args.send_seconds))]
            # the cap only catches a stuck server: arrivals end after ~send_seconds, requests take seconds
            res = await run_load(args.base_url, args.model, reqs, rate=args.rate,
                                 max_seconds=args.send_seconds + 120, scrape=False)
            s = summarize(res)
            totals["completed"] += s["completed"]
            totals["failed"] += s["failed"]
            round_errors = Counter(r.error.split(":")[0] for r in res["records"] if not r.ok)
            errors.update(round_errors)
            idle = None
            for _ in range(50):                                  # wait for the server to drain
                text = (await client.get(f"{args.base_url}/metrics")).text
                if not _metric(text, "num_requests_running") and not _metric(text, "num_requests_waiting"):
                    idle = text
                    break
                await asyncio.sleep(0.2)
            minute += 1
            point = {"minute": minute, "completed": s["completed"], "failed": s["failed"],
                     "errors": dict(round_errors), "idle": idle is not None}
            if idle:
                point["gpu_memory_allocated_bytes"] = _metric(idle, "gpu_memory_allocated_bytes")
                point["kv_free_blocks"] = _metric(idle, "kv_free_blocks")
            samples.append(point)
            print(json.dumps(point), flush=True)
    idle_pts = [p for p in samples if p["idle"]]
    ok = bool(idle_pts) and totals["failed"] == 0 and len(idle_pts) == len(samples)
    if idle_pts:
        mem0, free0 = idle_pts[0].get("gpu_memory_allocated_bytes"), idle_pts[0].get("kv_free_blocks")
        ok &= all(p.get("kv_free_blocks") == free0 for p in idle_pts)
        if mem0:
            ok &= all(abs((p.get("gpu_memory_allocated_bytes") or 0) - mem0) <= 64 * 2**20 for p in idle_pts)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "soak.json").write_text(json.dumps({"passed": ok, "totals": totals, "errors": dict(errors),
                                                   "samples": samples, "args": vars(args)}, indent=1))
    print("SOAK", "PASSED" if ok else "FAILED", totals, dict(errors))
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--send-seconds", type=float, default=45.0, help="arrivals per round, in seconds")
    raise SystemExit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
