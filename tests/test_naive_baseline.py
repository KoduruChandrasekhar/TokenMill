"""Baseline A (HF generate, one request at a time) speaks the same streaming format as the server,
serves requests one at a time, and releases its lock when a client disconnects."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from baselines.naive_hf import build_app
from infer.model.qwen2 import greedy_generate_reference


@pytest.fixture(scope="module")
def naive_server(tiny_model_dir):
    app = build_app(str(tiny_model_dir), "float32", "cpu", max_model_len=256)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def _chunks(text: str) -> list[dict]:
    events = [e[len("data: "):] for e in text.strip("\n").split("\n\n")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def test_completions_stream_matches_greedy(naive_server, tiny_model, tiny_tokenizer):
    prompt = "The quick brown fox"
    ids = tiny_tokenizer.encode(prompt, add_special_tokens=False)
    want = tiny_tokenizer.decode(greedy_generate_reference(tiny_model, ids, 10), skip_special_tokens=True)
    r = httpx.post(f"{naive_server}/v1/completions", json={"prompt": prompt, "max_tokens": 10, "temperature": 0,
                                                             "ignore_eos": True, "stream": True,
                                                             "stream_options": {"include_usage": True}})
    chunks = _chunks(r.text)
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 10
    assert "".join(c["choices"][0]["text"] for c in chunks[:-1]) == want
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"


def test_generate_endpoint(naive_server):
    r = httpx.post(f"{naive_server}/generate", json={"prompt": "hello", "max_tokens": 5})
    assert r.status_code == 200 and isinstance(r.json()["text"], str)
    r = httpx.post(f"{naive_server}/generate", json={"prompt": "hello", "max_tokens": 5, "stream": True})
    assert r.text.rstrip().endswith("data: [DONE]")


def test_one_at_a_time_and_disconnect_releases_lock(naive_server):
    with httpx.Client(timeout=60) as client:                     # abandon a long stream mid-way
        with client.stream("POST", f"{naive_server}/v1/completions",
                           json={"prompt": "abc", "max_tokens": 200, "ignore_eos": True, "stream": True}) as r:
            lines = r.iter_lines()
            for _ in range(4):
                next(lines)

    async def main():
        async with httpx.AsyncClient(timeout=60) as client:
            t0 = time.perf_counter()
            rs = await asyncio.gather(*(client.post(f"{naive_server}/v1/completions",
                                                    json={"prompt": "x", "max_tokens": 8, "ignore_eos": True})
                                        for _ in range(4)))
            return rs, time.perf_counter() - t0

    rs, elapsed = asyncio.run(main())                            # would hang if the lock leaked
    assert all(r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 8 for r in rs)
    assert elapsed < 30
