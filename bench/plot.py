"""Every figure and the README results table, rebuilt from raw JSON with one command.

    python -m bench.plot                      # reads bench/results, writes bench/results/figures + tables

Inputs (whatever exists is plotted; missing inputs are reported and skipped):
  raw/*.json                    one file per load_gen run (systems A-F, every load point, 3 runs)
  kv_experiments.json           concurrency vs KV budget, fragmentation, block sizes, recompute vs swap
  static_batching.json          throughput vs batch size, waste on the mixed workload
  profile/decode_breakdown_*.json  where one decode step's time goes
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bench.load_gen import summarize_raw  # noqa: E402

HERE = Path(__file__).resolve().parent

# Categorical slots in fixed order (validated palette, light mode); colour follows the system.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Lines can cross, so every pair drawn together must be distinguishable, not just neighbours in the
# slot order: {0, 2, 3, 4, 5} is the largest subset that passes the validator on all pairs, so the
# figures draw five systems (B-F; A is a single point, in the tables) with a marker each as well.
# The earlier graph versions keep E's colour and differ by line style.
SYSTEM_SLOT = {"D": 0, "C": 2, "B": 3, "F": 4, "E": 5, "E-pow2": 5, "E-ctx128": 5, "A": 1}
SYSTEM_STYLE = {"D": ("-", "o"), "C": ("-", "D"), "B": ("-", "v"), "F": ("-", "^"), "E": ("-", "s"),
                "E-pow2": (":", "x"), "E-ctx128": ("--", "+"), "A": ("-", "P")}
SYSTEM_NAME = {"A": "A naive HF", "B": "B static batching", "C": "C continuous, contiguous",
               "D": "D continuous, paged", "E": "E = D + CUDA graphs up to batch 32", "F": "F vLLM",
               "E-pow2": "graphs at every batch, power-of-two buckets",
               "E-ctx128": "graphs at every batch, 128-token buckets"}
TABLE_SYSTEMS = ["A", "B", "C", "D", "E-pow2", "E-ctx128", "E", "F"]
FIGURE_SYSTEMS = ["B", "C", "D", "E", "F"]
GRAPH_VERSIONS = ["D", "E-pow2", "E-ctx128", "E"]
# Tagged runs of D are experiments: each gets its own figure, baseline first (slot 0), variant second.
EXPERIMENTS = [("policy", "mixed", "prefill-first vs decode-first", ["D-prefill_first", "D-decode_first"]),
               ("chunked_prefill", "mixed", "chunked prefill (512-token budget)", ["D-unchunked", "D-chunked512"]),
               ("prefix_caching", "shared_prefix", "prefix caching, shared 1,000-token prefix",
                ["D-prefix_off", "D-prefix_on"]),
               ("preemption", "sharegpt", "preemption under overload (64 MiB KV)",
                ["D-overload_recompute", "D-overload_swap"])]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def color_for(label: str) -> str:
    """Fixed slot per system, so a system keeps its colour in every figure."""
    return SLOTS[SYSTEM_SLOT.get(label, 7)]


def line_kw(label: str) -> dict:
    ls, marker = SYSTEM_STYLE.get(label, ("-", "o"))
    return {"color": color_for(label), "linestyle": ls, "marker": marker, "ms": 6, "lw": 2,
            "label": SYSTEM_NAME.get(label, label)}


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {name}")


# --- raw run files ------------------------------------------------------------------------------

def load_runs(raw: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob("*.json")):
        d = json.loads(f.read_text())
        s = summarize_raw(d) if d.get("requests") else d["summary"]
        label = d["system"] + (f"-{d['tag']}" if d.get("tag") else "")
        v = d["load"]["value"]
        rows.append({
            "system": label, "workload": d["workload"], "load_type": d["load"]["type"],
            "load": math.inf if v == "inf" else float(v), "run": d["run"],
            "throughput": s["output_throughput_tok_s"], "req_throughput": s["request_throughput_req_s"],
            "goodput": s["goodput_req_s"], "ttft_p50": s["ttft_s"]["p50"], "ttft_p99": s["ttft_s"]["p99"],
            "tpot_p50": s["tpot_s"]["p50"], "tpot_p99": s["tpot_s"]["p99"], "itl_p99": s["itl_s"]["p99"],
            "e2e_p99": s["e2e_s"]["p99"], "max_concurrent": s.get("max_concurrent_seqs"),
            "saturated": s["saturated"], "gpu_minutes": d.get("gpu_minutes", 0.0), "file": f.name,
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["throughput", "req_throughput", "goodput", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
               "itl_p99", "e2e_p99", "max_concurrent"]
    g = df.groupby(["system", "workload", "load_type", "load"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["runs"] = g.size()
    out["saturated"] = g["saturated"].any()
    return out.reset_index()


def _xpos(loads: pd.Series, finite_max: float) -> pd.Series:
    return loads.where(loads.apply(math.isfinite), finite_max * 2)


def plot_experiments(agg: pd.DataFrame, out: Path) -> None:
    """One small-multiple figure per experiment: throughput, p99 TTFT and p99 TPOT vs rate."""
    rate = agg[agg.load_type == "rate"]
    for name, workload, title, labels in EXPERIMENTS:
        w = rate[(rate.workload == workload) & rate.system.isin(labels)]
        if w.system.nunique() < 2:
            continue
        finite = w.load[w.load.apply(math.isfinite)]
        fmax = finite.max() if len(finite) else 1.0
        ticks = sorted(set(w.load.apply(lambda v: v if math.isfinite(v) else fmax * 2)))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, col, ylabel, scale in ((axes[0], "throughput_mean", "output tokens / s", 1),
                                       (axes[1], "ttft_p99_mean", "p99 TTFT (ms)", 1e3),
                                       (axes[2], "tpot_p99_mean", "p99 TPOT (ms)", 1e3)):
            for slot, label in enumerate(labels):
                s = w[w.system == label].sort_values("load")
                ax.plot(_xpos(s.load, fmax), scale * s[col], marker="o", ms=6, lw=2, color=SLOTS[slot],
                        label=label.partition("-")[2])
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks, [f"{t:g}" if t <= fmax else "all" for t in ticks])
            if col == "ttft_p99_mean":
                ax.set_yscale("log")
            style(ax, ylabel, "requests per second", "")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"D: {title}, {workload} workload", x=0.01, ha="left", fontsize=11, color=INK)
        save(fig, out, f"9_experiment_{name}.png")


def plot_rate_figures(agg: pd.DataFrame, out: Path) -> None:
    agg = agg[agg.system.isin(FIGURE_SYSTEMS)]          # experiments and graph versions: own figures
    rate = agg[agg.load_type == "rate"]
    for workload in sorted(rate.workload.unique()):
        w = rate[rate.workload == workload]
        finite = w.load[w.load.apply(math.isfinite)]
        fmax = finite.max() if len(finite) else 1.0
        # 1. throughput vs request rate
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for system in sorted(w.system.unique()):
            s = w[w.system == system].sort_values("load")
            x = _xpos(s.load, fmax)
            ax.errorbar(x, s.throughput_mean, yerr=s.throughput_std.fillna(0), capsize=3, **line_kw(system))
        ax.set_xscale("log", base=2)
        ticks = sorted(set(finite.tolist())) + [fmax * 2]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks[:-1]] + ["all at once"])
        style(ax, f"Output throughput vs request rate ({workload})", "requests per second (Poisson)",
              "output tokens / s")
        ax.legend(frameon=False, fontsize=8)
        save(fig, out, f"1_throughput_vs_rate_{workload}.png")

        # 2. p99 TTFT vs throughput: lower and further right is better
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for system in sorted(w.system.unique()):
            s = w[w.system == system].sort_values("load")
            ax.plot(s.throughput_mean, 1e3 * s.ttft_p99_mean, **line_kw(system))
        ax.set_yscale("log")
        style(ax, f"Latency-throughput: p99 TTFT vs throughput ({workload})", "output tokens / s",
              "p99 time to first token (ms, log)")
        ax.legend(frameon=False, fontsize=8)
        save(fig, out, f"2_ttft_p99_vs_throughput_{workload}.png")

        # 3. p50 and p99 TPOT vs rate (two panels, one axis each)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, pct in zip(axes, ("p50", "p99")):
            for system in sorted(w.system.unique()):
                s = w[w.system == system].sort_values("load")
                ax.plot(_xpos(s.load, fmax), 1e3 * s[f"tpot_{pct}_mean"], **line_kw(system))
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks[:-1]] + ["all"])
            style(ax, f"{pct} TPOT ({workload})", "requests per second", "ms per output token")
        axes[0].legend(frameon=False, fontsize=8)
        save(fig, out, f"3_tpot_vs_rate_{workload}.png")

    conc = agg[agg.load_type == "concurrency"]
    for workload in sorted(conc.workload.unique()):
        w = conc[conc.workload == workload]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for system in sorted(w.system.unique()):
            s = w[w.system == system].sort_values("load")
            ax.plot(s.load, s.throughput_mean, **line_kw(system))
        users_axis(ax, w.load)
        style(ax, f"Throughput vs users in flight ({workload})", "concurrent users", "output tokens / s")
        ax.legend(frameon=False, fontsize=8)
        save(fig, out, f"1b_throughput_vs_concurrency_{workload}.png")


def users_axis(ax, loads: pd.Series) -> None:
    ax.set_xscale("log", base=2)
    ticks = sorted(loads.unique())
    ax.set_xticks(ticks, [f"{int(t)}" for t in ticks])
    ax.minorticks_off()


def plot_graph_versions(agg: pd.DataFrame, out: Path) -> None:
    """10. The three CUDA-graph versions against eager D, closed loop, both workloads."""
    conc = agg[(agg.load_type == "concurrency") & agg.system.isin(GRAPH_VERSIONS)]
    workloads = [w for w in ("fixed", "sharegpt") if w in set(conc.workload)]
    if not workloads or conc.system.nunique() < 2:
        return
    fig, axes = plt.subplots(1, len(workloads), figsize=(6 * len(workloads), 4), squeeze=False)
    for ax, workload in zip(axes[0], workloads):
        w = conc[conc.workload == workload]
        for system in [s for s in GRAPH_VERSIONS if s in set(w.system)]:
            s = w[w.system == system].sort_values("load")
            ax.plot(s.load, s.throughput_mean, **line_kw(system))
        users_axis(ax, w.load)
        style(ax, workload, "concurrent users", "output tokens / s")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("CUDA graphs: three versions vs eager decoding", x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, out, "10_cuda_graph_versions.png")


def plot_timeline(raw: Path, out: Path) -> None:
    """7. Batch size and KV usage over time during one run (the busiest D run)."""
    best, best_tp = None, -1.0
    for f in raw.glob("D_*.json"):
        d = json.loads(f.read_text())
        if d.get("timeline") and d["summary"]["output_throughput_tok_s"] > best_tp:
            best, best_tp = d, d["summary"]["output_throughput_tok_s"]
    if best is None:
        print("  skip timeline: no D runs with a metrics timeline")
        return
    tl = pd.DataFrame(best["timeline"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(tl.t, tl.get("running", 0), lw=2, color=SLOTS[0], label="running (batch size)")
    if "waiting" in tl:
        axes[0].plot(tl.t, tl.waiting, lw=2, color=SLOTS[1], label="waiting")
    axes[0].legend(frameon=False, fontsize=8)
    style(axes[0], f"Batch size over time: D, {best['workload']}, rate {best['load']['value']}", "", "requests")
    if "kv_usage" in tl:
        axes[1].plot(tl.t, 100 * tl.kv_usage, lw=2, color=SLOTS[2])
    style(axes[1], "KV cache usage", "seconds", "% of blocks")
    save(fig, out, "7_batch_and_kv_over_time.png")


def plot_kv(path: Path, out: Path) -> dict:
    if not path.exists():
        print(f"  skip KV figures: {path.name} missing (python -m bench.kv_experiments)")
        return {}
    d = json.loads(path.read_text())
    c = pd.DataFrame(d["concurrency"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cache, slot in (("paged", 0), ("contiguous", 2)):
        s = c[c.cache_type == cache].sort_values("kv_cache_mb")
        ax.plot(s.kv_cache_mb, s.peak_running, marker="o", ms=6, lw=2, color=SLOTS[slot], label=cache)
        last = s.iloc[-1]
        ax.annotate(f"{int(last.peak_running)}", (last.kv_cache_mb, last.peak_running), textcoords="offset points",
                    xytext=(6, 0), color=INK, fontsize=9, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(c.kv_cache_mb.unique()), [f"{int(x)}" for x in sorted(c.kv_cache_mb.unique())])
    style(ax, "Max concurrent sequences vs KV memory", "KV cache budget (MiB)", "peak running sequences")
    ax.legend(frameon=False)
    save(fig, out, "4_max_concurrency_vs_kv_budget.png")

    b = pd.DataFrame(d["block_sizes"]).sort_values("block_size")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    labels = [str(x) for x in b.block_size]
    for ax, col, title, unit, scale in ((axes[0], "fragmentation", "KV waste", "% allocated but unused", 100),
                                        (axes[1], "mean_decode_step_ms", "Decode step time", "ms", 1),
                                        (axes[2], "throughput_tok_s", "Throughput", "output tokens / s", 1)):
        ax.bar(labels, scale * b[col], color=SLOTS[0], width=0.55)
        for x, y in zip(labels, scale * b[col]):
            ax.annotate(f"{y:.1f}" if y < 100 else f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color=INK)
        style(ax, title, "block size (tokens)", unit)
    save(fig, out, "5_block_size_ablation.png")

    headline = {}
    for budget in sorted(c.kv_cache_mb.unique()):
        p = c[(c.kv_cache_mb == budget) & (c.cache_type == "paged")].peak_running
        q = c[(c.kv_cache_mb == budget) & (c.cache_type == "contiguous")].peak_running
        if len(p) and len(q) and q.iloc[0]:
            headline[f"concurrency_gain_{int(budget)}MiB"] = float(p.iloc[0] / q.iloc[0])
    headline["fragmentation"] = {f"{r.cache_type}_{int(r.kv_cache_mb)}MiB": r.fragmentation for r in c.itertuples()}
    headline["overload"] = d.get("overload", [])
    for o in headline["overload"]:
        o.pop("lengths", None)
    return headline


def plot_static(path: Path, out: Path) -> dict:
    if not path.exists():
        print(f"  skip static-batching figure: {path.name} missing (python -m bench.static_experiments)")
        return {}
    d = json.loads(path.read_text())
    curve = pd.DataFrame(d["throughput_vs_batch"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(curve.batch_size, curve.throughput_tok_s, marker="o", ms=6, lw=2, color=SLOTS[3])
    ax.set_xscale("log", base=2)
    ax.set_xticks(curve.batch_size, [str(b) for b in curve.batch_size])
    w = d["mixed_static"]["waste"]
    style(ax, f"Static batching: throughput vs batch size (mixed-length waste {100 * w:.0f}%)",
          "batch size", "output tokens / s")
    save(fig, out, "8_static_throughput_vs_batch.png")
    return {"static_waste_mixed": w, "static_mixed_tok_s": d["mixed_static"]["throughput_tok_s"],
            "continuous_mixed_tok_s": d["mixed_continuous"]["throughput_tok_s"]}


def plot_profile(prof_dir: Path, out: Path) -> None:
    files = sorted(prof_dir.glob("decode_breakdown_*.json")) if prof_dir.exists() else []
    if not files:
        print("  skip decode breakdown: no profile (python -m bench.profile_decode)")
        return
    fig, ax = plt.subplots(figsize=(8, 2.0 + 1.1 * len(files)))
    cats: list[str] = []
    docs = [json.loads(f.read_text()) for f in files]
    for d in docs:
        for k in d["ms_per_step_by_category"]:
            if k not in cats:
                cats.append(k)
    for i, d in enumerate(docs):
        left = 0.0
        for j, k in enumerate(cats):
            v = d["ms_per_step_by_category"].get(k, 0.0)
            ax.barh(i, v, left=left, color=SLOTS[j % len(SLOTS)], edgecolor="white", linewidth=2,
                    label=k if i == 0 else None, height=0.6)
            left += v
    ax.set_yticks(range(len(docs)), [f"batch {d['batch']}{' graphs' if d.get('cuda_graphs') else ''}"
                                     for d in docs])
    style(ax, "Where one decode step's time goes", "ms per decode step", "")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    save(fig, out, "6_decode_step_breakdown.png")


# --- tables ---------------------------------------------------------------------------------------

def _fmt(mean, std=None, scale=1.0, digits=0) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "–"
    s = f"{scale * mean:.{digits}f}"
    if std is not None and not (isinstance(std, float) and math.isnan(std)) and std > 0:
        s += f" ± {scale * std:.{digits}f}"
    return s


def results_table(agg: pd.DataFrame, headline: dict) -> str:
    """Fixed workload (512 in / 128 out). Throughput and max concurrency at "all at once";
    latencies at 4 req/s (A only runs all at once, so its latencies are from that point)."""
    fixed = agg[(agg.workload == "fixed") & (agg.load_type == "rate")]
    lines = ["| System | Output tok/s | p50 TTFT (ms) | p99 TTFT (ms) | p50 TPOT (ms) | Max concurrent seqs |",
             "|---|---|---|---|---|---|"]
    peak = {}
    for system in TABLE_SYSTEMS:
        s = fixed[fixed.system == system]
        if s.empty:
            if "-" not in system:
                lines.append(f"| {SYSTEM_NAME[system]} | not run | | | | |")
            continue
        top = s[~s.load.apply(math.isfinite)]
        top = top.iloc[0] if len(top) else s.sort_values("throughput_mean").iloc[-1]
        lat = s[s.load == 4.0]
        lat = lat.iloc[0] if len(lat) else top
        peak[system] = top.throughput_mean
        lines.append(f"| {SYSTEM_NAME[system]} | {_fmt(top.throughput_mean, top.throughput_std)} | "
                     f"{_fmt(lat.ttft_p50_mean, scale=1e3)} | {_fmt(lat.ttft_p99_mean, scale=1e3)} | "
                     f"{_fmt(lat.tpot_p50_mean, scale=1e3, digits=1)} | {_fmt(top.max_concurrent_mean)} |")
    if "D" in peak and "B" in peak:
        headline["D_vs_static_throughput_x"] = peak["D"] / peak["B"]
    if "D" in peak and "F" in peak:
        headline["D_pct_of_vllm"] = 100 * peak["D"] / peak["F"]
    return "\n".join(lines)


def concurrency_table(agg: pd.DataFrame, workload: str, headline: dict) -> str:
    """Closed loop: N users, each sends its next request when the last one finishes. Same N for
    every server, so unlike "all at once" this does not depend on each server's max_num_seqs."""
    w = agg[(agg.workload == workload) & (agg.load_type == "concurrency")]
    if w.empty:
        return ""
    loads = sorted(w.load.unique())
    lines = ["| System | " + " | ".join(f"{int(n)} user{'s' if n != 1 else ''}: tok/s, p50 TPOT"
                                         for n in loads) + " |",
             "|---|" + "---|" * len(loads)]
    for system in [s for s in TABLE_SYSTEMS if s in set(w.system)]:
        cells = []
        for n in loads:
            r = w[(w.system == system) & (w.load == n)]
            cells.append("–" if r.empty else f"{_fmt(r.iloc[0].throughput_mean)}, "
                         f"{_fmt(r.iloc[0].tpot_p50_mean, scale=1e3, digits=1)} ms")
        lines.append(f"| {SYSTEM_NAME.get(system, system)} | " + " | ".join(cells) + " |")
    for system in ("D", "E"):
        for n in loads:
            ours = w[(w.system == system) & (w.load == n)]
            ref = w[(w.system == "F") & (w.load == n)]
            if len(ours) and len(ref):
                headline[f"{system}_pct_of_vllm_{workload}_{int(n)}users"] = (
                    100 * ours.iloc[0].throughput_mean / ref.iloc[0].throughput_mean)
    return "\n".join(lines)


def sweep_table(agg: pd.DataFrame, workload: str) -> str:
    w = agg[(agg.workload == workload) & (agg.load_type == "rate")]
    if w.empty:
        return ""
    lines = ["| System | rate (req/s) | tok/s | p50 TTFT (ms) | p99 TTFT (ms) | p50 TPOT (ms) | p99 TPOT (ms) | "
             "goodput (req/s) |", "|---|---|---|---|---|---|---|---|"]
    for r in w.sort_values(["system", "load"]).itertuples():
        lines.append(f"| {r.system} | {'all' if not math.isfinite(r.load) else f'{r.load:g}'} | "
                     f"{_fmt(r.throughput_mean, r.throughput_std)} | {_fmt(r.ttft_p50_mean, scale=1e3)} | "
                     f"{_fmt(r.ttft_p99_mean, scale=1e3)} | {_fmt(r.tpot_p50_mean, scale=1e3, digits=1)} | "
                     f"{_fmt(r.tpot_p99_mean, scale=1e3, digits=1)} | {_fmt(r.goodput_mean, digits=2)}"
                     f"{' (saturated)' if r.saturated else ''} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    res = Path(args.results_dir)
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    raw = res / "raw"
    headline: dict = {}
    md = ["# Results", "", "Generated by `python -m bench.plot` from the raw JSON in this directory.", ""]
    df = load_runs(raw) if raw.exists() else pd.DataFrame()
    if df.empty:
        print("  no raw load_gen runs found")
    else:
        agg = aggregate(df)
        agg.to_csv(res / "summary.csv", index=False)
        plot_rate_figures(agg, out)
        plot_experiments(agg, out)
        plot_graph_versions(agg, out)
        plot_timeline(raw, out)
        md += ["## Fixed workload (512 in, 128 out)", "",
               "Throughput and max concurrent sequences at \"all at once\"; latencies at 4 req/s "
               "(mean ± std over runs).", "", results_table(agg, headline), ""]
        for workload in ("fixed", "sharegpt"):
            t = concurrency_table(agg, workload, headline)
            if t:
                md += [f"## Users in flight: {workload}", "", t, ""]
        for workload in ("sharegpt", "mixed", "fixed", "shared_prefix"):
            t = sweep_table(agg, workload)
            if t:
                md += [f"## Rate sweep: {workload}", "", t, ""]
        headline["gpu_minutes_total"] = float(df.gpu_minutes.sum())
    headline.update(plot_kv(res / "kv_experiments.json", out))
    headline.update(plot_static(res / "static_batching.json", out))
    plot_profile(res / "profile", out)
    (res / "results.md").write_text("\n".join(md), encoding="utf-8")
    (res / "headline.json").write_text(json.dumps(headline, indent=1, default=str))
    print(f"wrote {res / 'results.md'} and {res / 'headline.json'}")


if __name__ == "__main__":
    main()
