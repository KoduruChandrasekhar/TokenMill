"""Terminal demo: N concurrent streams with live progress and the server's /metrics.

    python -m bench.demo --base-url http://localhost:8000 --streams 50
    python -m bench.demo --streams 50 --record demo_frames.jsonl --quiet    # then:
    python -m bench.render_gif demo_frames.jsonl docs/demo.gif
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time

import httpx

PROMPTS = ["Explain how {} works.", "Write a short poem about {}.", "List five facts about {}.",
           "Summarize the history of {}.", "Give three tips for learning {}."]
TOPICS = ["GPUs", "the ocean", "Rust", "jazz", "volcanoes", "chess", "bread", "satellites", "bees", "Tokyo"]


async def stream(client, url, i, state):
    prompt = PROMPTS[i % len(PROMPTS)].format(TOPICS[i % len(TOPICS)])
    max_tokens = 64 + 8 * (i % 12)
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "seed": i, "stream": True}
    state[i] = {"n": 0, "text": "", "done": False, "max": max_tokens}
    async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                ch = json.loads(line[6:])["choices"]
                if ch and ch[0]["delta"].get("content"):
                    state[i]["n"] += 1
                    state[i]["text"] = (state[i]["text"] + ch[0]["delta"]["content"]).replace("\n", " ")[-48:]
    state[i]["done"] = True


def frame(state, head, t) -> list[str]:
    done = sum(s["done"] for s in state.values())
    lines = [f"t={t:5.1f}s  {head}", f"{done}/{len(state)} streams finished", ""]
    for i in sorted(state)[:24]:
        s = state[i]
        bar = "#" * round(24 * min(1.0, s["n"] / s["max"]))
        lines.append(f"{i:3d} [{bar:24s}] {s['n']:3d} {'done' if s['done'] else '    '} {s['text'][-44:]}")
    return lines


async def render(client, url, state, n, t0, record=None, quiet=False):
    while True:
        finished = len(state) == n and all(s["done"] for s in state.values())
        try:
            m = (await client.get(f"{url}/metrics")).text

            def g(k: str, text: str = m) -> float:
                hit = re.search(rf"^{k} (\S+)", text, re.M)
                return float(hit.group(1)) if hit else 0.0
            head = (f"running {int(g('num_requests_running')):3d}  waiting {int(g('num_requests_waiting')):3d}  "
                    f"KV {100 * g('kv_cache_usage_ratio'):5.1f}%  tokens {int(g('generation_tokens_total')):6d}")
        except Exception:
            head = ""
        t = time.time() - t0
        lines = frame(state, head, t)
        if record:
            record.write(json.dumps({"t": t, "lines": lines}) + "\n")
        if not quiet:
            sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
            sys.stdout.flush()
        if finished:
            return
        await asyncio.sleep(0.25)


async def main_async(args):
    state: dict = {}
    record = open(args.record, "w", encoding="utf-8") if args.record else None
    async with httpx.AsyncClient(timeout=300) as client:
        t0 = time.time()
        await asyncio.gather(render(client, args.base_url, state, args.streams, t0, record, args.quiet),
                             *(stream(client, args.base_url, i, state) for i in range(args.streams)))
    if record:
        record.close()
    print(f"{args.streams} streams finished in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--streams", type=int, default=50)
    ap.add_argument("--record", default=None, help="write every screen as a JSON line (for render_gif)")
    ap.add_argument("--quiet", action="store_true", help="do not draw to the terminal")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
