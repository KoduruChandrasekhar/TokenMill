"""Async load generator: Poisson arrivals or fixed concurrency against any OpenAI-compatible
/v1/completions endpoint (this server, the baselines, vLLM). Records a timestamp for every
streamed chunk and writes one raw JSON file per run.

    python -m bench.load_gen --base-url http://localhost:8000 --system D --workload fixed \
        --num-requests 300 --rates 1,2,4,8,16,32,inf --concurrency 1,8,32,128 --runs 3

Metrics (p50/p90/p99 of each latency):
  TTFT  send -> first token            TPOT  (E2E - TTFT) / (output tokens - 1)
  ITL   every gap between chunks       E2E   send -> last token
  output throughput = output tokens / duration;  goodput = requests/s meeting the SLO.
The SLO is fixed up front: TTFT <= 1 s and TPOT <= 100 ms.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import httpx
import numpy as np

SLO_TTFT_S = 1.0
SLO_TPOT_S = 0.100
SATURATED_P99_TTFT_S = 30.0
MAX_RUN_SECONDS = 15 * 60
RESULTS = Path(__file__).resolve().parent / "results" / "raw"


@dataclass
class RequestRecord:
    prompt_len: int
    output_len: int                       # requested
    send_time: float = 0.0
    chunk_times: list[float] = field(default_factory=list)
    end_time: float = 0.0
    output_tokens: int = 0                # from usage (falls back to chunk count)
    ok: bool = False
    status: int = 0
    error: str = ""

    @property
    def ttft(self) -> float | None:
        return self.chunk_times[0] - self.send_time if self.chunk_times else None

    @property
    def e2e(self) -> float:
        return self.end_time - self.send_time

    @property
    def tpot(self) -> float | None:
        if self.output_tokens < 2 or not self.chunk_times:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)

    @property
    def itls(self) -> list[float]:
        return list(np.diff(self.chunk_times)) if len(self.chunk_times) > 1 else []


async def send_one(client: httpx.AsyncClient, url: str, model: str, req, rec: RequestRecord) -> None:
    payload = {"model": model, "prompt": req.prompt, "max_tokens": req.output_len, "temperature": 0.0,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    rec.send_time = time.perf_counter()
    try:
        async with client.stream("POST", url, json=payload) as resp:
            rec.status = resp.status_code
            if resp.status_code != 200:
                rec.error = (await resp.aread()).decode(errors="replace")[:300]
                rec.end_time = time.perf_counter()
                return
            n_chunks = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    rec.error = str(obj["error"])[:300]
                    break
                if obj.get("usage"):
                    rec.output_tokens = obj["usage"].get("completion_tokens", rec.output_tokens)
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    rec.chunk_times.append(time.perf_counter())
                    n_chunks += 1
            rec.end_time = rec.chunk_times[-1] if rec.chunk_times else time.perf_counter()
            if not rec.output_tokens:
                rec.output_tokens = n_chunks
            rec.ok = not rec.error and rec.output_tokens > 0
    except Exception as e:  # connection reset, timeout, ...
        rec.error = f"{type(e).__name__}: {e}"[:300]
        rec.end_time = time.perf_counter()


async def scrape_metrics(client: httpx.AsyncClient, base_url: str, t0: float, out: list, stop: asyncio.Event):
    """Sample the server's /metrics twice a second: batch size and KV usage over time."""
    names = {"running": r"^(?:vllm:)?num_requests_running(?:\{[^}]*\})? (\S+)",
             "waiting": r"^(?:vllm:)?num_requests_waiting(?:\{[^}]*\})? (\S+)",
             "kv_usage": r"^(?:vllm:)?(?:kv_cache_usage_ratio|kv_cache_usage_perc|gpu_cache_usage_perc)(?:\{[^}]*\})? (\S+)"}
    while not stop.is_set():
        try:
            text = (await client.get(f"{base_url}/metrics", timeout=2)).text
            point = {"t": time.perf_counter() - t0}
            for key, pat in names.items():
                m = re.search(pat, text, re.M)
                if m:
                    point[key] = float(m.group(1))
            out.append(point)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), 0.5)
        except TimeoutError:
            pass


async def run_load(base_url: str, model: str, requests: list, rate: float | None = None,
                   concurrency: int | None = None, seed: int = 0, max_seconds: float = MAX_RUN_SECONDS,
                   scrape: bool = True) -> dict:
    """rate=inf (or None with no concurrency) sends everything at once."""
    url = f"{base_url}/v1/completions"
    records = [RequestRecord(r.prompt_len, r.output_len) for r in requests]
    rng = np.random.default_rng(seed)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeline: list = []
    stop = asyncio.Event()
    async with httpx.AsyncClient(timeout=httpx.Timeout(max_seconds + 60), limits=limits) as client:
        t0 = time.perf_counter()
        scraper = asyncio.create_task(scrape_metrics(client, base_url, t0, timeline, stop)) if scrape else None
        sem = asyncio.Semaphore(concurrency) if concurrency else None

        async def guarded(req, rec):
            if sem is None:
                return await send_one(client, url, model, req, rec)
            async with sem:
                return await send_one(client, url, model, req, rec)

        tasks = []
        for req, rec in zip(requests, records):
            tasks.append(asyncio.create_task(guarded(req, rec)))
            if rate and math.isfinite(rate) and not concurrency:
                await asyncio.sleep(float(rng.exponential(1.0 / rate)))
        done, pending = await asyncio.wait(tasks, timeout=max(0.0, max_seconds - (time.perf_counter() - t0)))
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        duration = time.perf_counter() - t0
        stop.set()
        if scraper:
            await scraper
    for rec in records:
        if not rec.ok and not rec.error:
            rec.error = "cancelled: run hit the time cap"
    return {"records": records, "duration": duration, "hit_time_cap": bool(pending), "timeline": timeline}


def _pcts(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "p50": None, "p90": None, "p99": None, "n": 0}
    a = np.asarray(values)
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)), "n": len(a)}


def summarize(result: dict) -> dict:
    recs: list[RequestRecord] = result["records"]
    ok = [r for r in recs if r.ok]
    duration = result["duration"]
    if ok:                                 # measure from first send to last token
        # requests cancelled at the time cap while still queued were never sent (send_time 0)
        duration = max(r.end_time for r in ok) - min(r.send_time for r in recs if r.send_time)
    out_tokens = sum(r.output_tokens for r in ok)
    good = [r for r in ok if r.ttft is not None and r.ttft <= SLO_TTFT_S and (r.tpot or 0) <= SLO_TPOT_S]
    ttft = _pcts([r.ttft for r in ok if r.ttft is not None])
    # None when the server exports no running gauge (the naive baseline), not 0
    peak_running = max((p["running"] for p in result["timeline"] if "running" in p), default=None)
    return {
        "completed": len(ok), "failed": len(recs) - len(ok),
        "rejected_503": sum(1 for r in recs if r.status == 503),
        "duration_s": duration,
        "output_throughput_tok_s": out_tokens / duration if duration else 0.0,
        "request_throughput_req_s": len(ok) / duration if duration else 0.0,
        "goodput_req_s": len(good) / duration if duration else 0.0,
        "total_output_tokens": out_tokens,
        "ttft_s": ttft, "tpot_s": _pcts([r.tpot for r in ok if r.tpot is not None]),
        "itl_s": _pcts([x for r in ok for x in r.itls]), "e2e_s": _pcts([r.e2e for r in ok]),
        "max_concurrent_seqs": peak_running,
        # Saturation is queueing (p99 TTFT), not run length: one user in flight at a time can hit
        # the time cap while the server is idle most of the run.
        "saturated": (ttft["p99"] or 0) > SATURATED_P99_TTFT_S,
        "hit_time_cap": result["hit_time_cap"],
        "slo": {"ttft_s": SLO_TTFT_S, "tpot_s": SLO_TPOT_S},
    }


def summarize_raw(doc: dict) -> dict:
    """Recompute a raw run file's summary from its stored per-request timestamps, so files written
    by an older load_gen pick up fixes to the metric definitions."""
    names = {f.name for f in fields(RequestRecord)}
    recs = [RequestRecord(**{k: v for k, v in r.items() if k in names}) for r in doc["requests"]]
    capped = any(r.error.startswith("cancelled") for r in recs)
    return summarize({"records": recs, "duration": doc["summary"]["duration_s"], "hit_time_cap": capped,
                      "timeline": doc.get("timeline", [])})


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "time": time.time()}
    for mod in ("torch", "vllm", "transformers", "httpx"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for key, cmd in {"git_commit": ["git", "rev-parse", "HEAD"],
                     "nvidia_smi": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                                    "--format=csv,noheader"]}.items():
        try:
            env[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            env[key] = None
    # Without git (the Kaggle zip), scripts/make_kaggle_zip.py's digest identifies the code instead.
    digest = Path(__file__).resolve().parents[1] / "SOURCE_DIGEST"
    env["source_digest"] = digest.read_text().strip() if digest.exists() else None
    return env


def _parse_list(s: str | None, conv=float) -> list:
    return [conv(x) for x in s.split(",")] if s else []


async def warmup(base_url: str, model: str, requests: list, n: int = 20) -> None:
    """20 requests, results discarded: first-call compilation, allocator growth, connection setup."""
    if n <= 0:
        return
    await run_load(base_url, model, requests[:n], rate=None, concurrency=8, scrape=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--system", required=True, help="label, e.g. A, B, C, D, E, F")
    ap.add_argument("--workload", default="fixed", choices=["fixed", "mixed", "sharegpt", "shared_prefix"])
    ap.add_argument("--num-requests", type=int, default=300)
    ap.add_argument("--rates", default=None, help="comma list of req/s; inf = all at once")
    ap.add_argument("--concurrency", default=None, help="comma list of users in flight")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=MAX_RUN_SECONDS)
    ap.add_argument("--sharegpt-path", default=None)
    ap.add_argument("--input-len", type=int, default=None, help="fixed workload: prompt tokens (default 512)")
    ap.add_argument("--output-len", type=int, default=None, help="fixed workload: output tokens (default 128)")
    ap.add_argument("--tag", default="", help="extra label stored with results (e.g. decode_first)")
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--no-stop-on-saturation", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from bench.datasets import load_workload

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    kw = {"path": args.sharegpt_path} if args.workload == "sharegpt" else {}
    if args.workload == "fixed":
        kw.update({k: v for k, v in (("input_len", args.input_len), ("output_len", args.output_len)) if v})
    requests = load_workload(args.workload, tok, args.num_requests, seed=args.seed, **kw)
    warm = load_workload("fixed", tok, args.warmup, seed=999, input_len=64, output_len=16) if args.warmup else []
    env = environment()
    try:                                       # vLLM serves /version; our server 404s harmlessly
        env["server_version"] = httpx.get(f"{args.base_url}/version", timeout=5).text[:200]
    except Exception:
        env["server_version"] = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = [("rate", r) for r in _parse_list(args.rates)] + [("concurrency", int(c)) for c in
                                                               _parse_list(args.concurrency)]
    if not points:
        points = [("rate", math.inf)]

    async def go():
        await warmup(args.base_url, args.model, warm, args.warmup)
        total_gpu_min = 0.0
        saturated_kind = set()
        for kind, value in points:
            # Saturated: higher rates only add queueing. "All at once" still runs: it is the
            # peak-throughput point.
            all_at_once = kind == "rate" and not math.isfinite(value)
            if kind in saturated_kind and not all_at_once and not args.no_stop_on_saturation:
                print(f"skip {kind}={value}: saturated at a lower load")
                continue
            for run in range(args.runs):
                res = await run_load(args.base_url, args.model, requests,
                                     rate=value if kind == "rate" else None,
                                     concurrency=value if kind == "concurrency" else None,
                                     seed=args.seed + run, max_seconds=args.max_seconds)
                summ = summarize(res)
                gpu_min = res["duration"] / 60
                total_gpu_min += gpu_min
                label = "inf" if kind == "rate" and not math.isfinite(value) else str(value)
                doc = {"system": args.system, "tag": args.tag, "workload": args.workload,
                       "num_requests": len(requests), "load": {"type": kind, "value": label}, "run": run,
                       "summary": summ, "gpu_minutes": gpu_min, "env": env, "args": vars(args),
                       "timeline": res["timeline"],
                       "requests": [{**asdict(r), "ttft": r.ttft, "tpot": r.tpot} for r in res["records"]]}
                name = f"{args.system}{'-' + args.tag if args.tag else ''}_{args.workload}_{kind}-{label}_run{run}.json"
                (out_dir / name).write_text(json.dumps(doc))
                print(f"[{args.system}{' ' + args.tag if args.tag else ''}] {args.workload} {kind}={label} run {run}: "
                      f"{summ['output_throughput_tok_s']:.0f} tok/s, TTFT p50 {1e3 * (summ['ttft_s']['p50'] or 0):.0f} ms "
                      f"p99 {1e3 * (summ['ttft_s']['p99'] or 0):.0f} ms, TPOT p50 {1e3 * (summ['tpot_s']['p50'] or 0):.1f} ms, "
                      f"ok {summ['completed']}/{len(requests)}, {gpu_min:.1f} GPU-min"
                      f"{' SATURATED' if summ['saturated'] else ''}", flush=True)
                if summ["saturated"]:
                    saturated_kind.add(kind)
        ledger = out_dir / "gpu_minutes.jsonl"
        with ledger.open("a") as f:
            f.write(json.dumps({"system": args.system, "tag": args.tag, "workload": args.workload,
                                "gpu_minutes": total_gpu_min, "time": time.time()}) + "\n")
        print(f"total {total_gpu_min:.1f} GPU-minutes for this invocation")

    asyncio.run(go())


if __name__ == "__main__":
    main()
