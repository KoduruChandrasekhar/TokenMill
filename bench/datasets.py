"""Benchmark workloads. Every request carries an exact output length (sent with ignore_eos), so
every system generates the same number of tokens and an early stop never looks like speed.

* sharegpt:      1,000 real conversations; first user turn = prompt, real reply length = max_tokens.
                 Fixed seed; prompts over 1,024 tokens dropped (the workload vLLM's own benchmarks use).
* fixed:         512 input tokens, 128 output tokens. Clean for comparing systems.
* mixed:         half the requests want 32 tokens, half want 512: where static batching wastes most.
* shared_prefix: every request starts with the same ~1,000-token system prompt (prefix caching).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

_WORDS = (
    "time person year way day thing man world life hand part child eye woman place work week case point "
    "government company number group problem fact be have do say get make go know take see come think look "
    "want give use find tell ask seem feel try leave call good new first last long great little own other old "
    "right big high different small large next early young important few public bad same able system program "
    "question during market history data model memory network server request token cache block batch stream "
    "engine kernel layer attention vector matrix compute speed latency throughput river mountain garden "
    "music story letter morning evening window table city country language science water light sound color "
    "paper money family friend teacher student school book door road car train plane ship energy power"
).split()


@dataclass
class BenchRequest:
    prompt: str
    prompt_len: int
    output_len: int

    def to_dict(self) -> dict:
        return asdict(self)


def text_of_length(tokenizer, n: int, rng: random.Random) -> str:
    """Random English words whose tokenization is exactly n tokens."""
    for _ in range(100):
        words = " ".join(rng.choice(_WORDS) for _ in range(n + 8))
        ids = tokenizer.encode(words, add_special_tokens=False)
        text = tokenizer.decode(ids[:n])
        if len(tokenizer.encode(text, add_special_tokens=False)) == n:   # the cut re-tokenizes cleanly
            return text
    raise RuntimeError(f"could not build a {n}-token prompt")


def fixed(tokenizer, n: int, input_len: int = 512, output_len: int = 128, seed: int = 0) -> list[BenchRequest]:
    rng = random.Random(seed)
    return [BenchRequest(text_of_length(tokenizer, input_len, rng), input_len, output_len) for _ in range(n)]


def mixed(tokenizer, n: int, input_len: int = 256, short: int = 32, long: int = 512,
          seed: int = 0) -> list[BenchRequest]:
    rng = random.Random(seed)
    outs = [short] * (n // 2) + [long] * (n - n // 2)
    rng.shuffle(outs)
    return [BenchRequest(text_of_length(tokenizer, input_len, rng), input_len, o) for o in outs]


def shared_prefix(tokenizer, n: int, prefix_len: int = 1000, suffix_len: int = 32, output_len: int = 64,
                  seed: int = 0) -> list[BenchRequest]:
    rng = random.Random(seed)
    prefix = "You are a helpful assistant. " + text_of_length(tokenizer, prefix_len - 7, rng)
    reqs = []
    for _ in range(n):
        prompt = prefix + "\n\nQuestion: " + text_of_length(tokenizer, suffix_len, rng)
        reqs.append(BenchRequest(prompt, len(tokenizer.encode(prompt, add_special_tokens=False)), output_len))
    return reqs


def sharegpt(tokenizer, n: int = 1000, seed: int = 0, max_prompt_len: int = 1024, max_model_len: int = 2048,
             path: str | None = None) -> list[BenchRequest]:
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    convs = [d["conversations"] for d in data if len(d.get("conversations", [])) >= 2]
    rng = random.Random(seed)
    rng.shuffle(convs)
    out: list[BenchRequest] = []
    for conv in convs:
        if conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
            continue
        prompt, reply = conv[0]["value"], conv[1]["value"]
        p = len(tokenizer.encode(prompt, add_special_tokens=False))
        o = len(tokenizer.encode(reply, add_special_tokens=False))
        if p < 4 or o < 4 or p > max_prompt_len or p + o > max_model_len:   # same filters as vLLM
            continue
        out.append(BenchRequest(prompt, p, o))
        if len(out) == n:
            break
    return out


WORKLOADS = {"fixed": fixed, "mixed": mixed, "shared_prefix": shared_prefix, "sharegpt": sharegpt}


def load_workload(name: str, tokenizer, n: int, seed: int = 0, **kw) -> list[BenchRequest]:
    reqs = WORKLOADS[name](tokenizer, n, seed=seed, **kw)
    if len(reqs) < n:
        raise ValueError(f"workload {name} produced {len(reqs)} < {n} requests")
    return reqs[:n]
