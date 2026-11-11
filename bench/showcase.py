"""Export the measured results for the showcase page (docs/showcase/index.html).

    python -m bench.showcase            # after python -m bench.plot; writes docs/showcase/data.js

The page is static HTML that reads window.SHOWCASE from data.js, so it opens straight from disk
and every number on it comes from bench/results.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "showcase"


def _num(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), 6)


def main() -> None:
    agg = pd.read_csv(RES / "summary.csv")
    rows = []
    for r in agg.itertuples():
        rows.append({"system": r.system, "workload": r.workload, "type": r.load_type,
                     "load": "all" if not math.isfinite(r.load) else _num(r.load),
                     "tok_s": _num(r.throughput_mean), "tok_s_std": _num(r.throughput_std),
                     "ttft_p50": _num(r.ttft_p50_mean), "ttft_p99": _num(r.ttft_p99_mean),
                     "tpot_p50": _num(r.tpot_p50_mean), "tpot_p99": _num(r.tpot_p99_mean),
                     "goodput": _num(r.goodput_mean), "runs": int(r.runs)})
    load = lambda name: json.loads((RES / name).read_text())      # noqa: E731
    kv, static, batch1, soak = (load(n) for n in ("kv_experiments.json", "static_batching.json",
                                                  "batch1.json", "soak.json"))
    profile = {p.stem.removeprefix("decode_breakdown_"): {k: v for k, v in load(f"profile/{p.name}").items()
                                                          if k in ("batch", "cuda_graphs", "wall_ms_per_step",
                                                                   "ms_per_step_by_category")}
               for p in sorted((RES / "profile").glob("decode_breakdown_*.json"))}
    data = {
        "rows": rows,
        "headline": load("headline.json"),
        "kv_concurrency": [{k: c[k] for k in ("cache_type", "kv_cache_mb", "peak_running", "fragmentation")}
                           for c in kv["concurrency"]],
        "block_sizes": [{k: b[k] for k in ("block_size", "fragmentation", "mean_decode_step_ms", "throughput_tok_s")}
                        for b in kv["block_sizes"]],
        "static_curve": static["throughput_vs_batch"],
        "static_mixed": {"waste": static["mixed_static"]["waste"],
                         "static_tok_s": static["mixed_static"]["throughput_tok_s"],
                         "continuous_tok_s": static["mixed_continuous"]["throughput_tok_s"]},
        "batch1": batch1["results"],
        "profile": profile,
        "soak": {"passed": soak["passed"], "totals": soak["totals"], "rounds": len(soak["samples"]),
                 "memory": sorted({s["gpu_memory_allocated_bytes"] for s in soak["samples"]}),
                 "free_blocks": sorted({s["kv_free_blocks"] for s in soak["samples"]})},
        "fp16_tolerance": json.loads((ROOT / "tests" / "fp16_tolerance.json").read_text())["tolerance"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.js").write_text("window.SHOWCASE = " + json.dumps(data, separators=(",", ":")) + ";\n",
                                 encoding="utf-8")
    shutil.copyfile(RES / "demo.gif", OUT / "demo.gif")
    print(f"wrote {OUT / 'data.js'} ({len(rows)} load points)")


if __name__ == "__main__":
    main()
