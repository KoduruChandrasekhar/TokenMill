"""API contract against a live server (tiny model): SSE format, [DONE], stop strings, finish_reason,
usage counts, 503 when full, 400 when too long, abort on disconnect, OpenAI SDK, 100 streams."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from helpers import tiny_cfg
from infer.api.server import create_app
from infer.engine.engine import AsyncEngine, LLMEngine
from infer.engine.static_engine import StaticBatchEngine
from infer.metrics import Metrics
from infer.model.qwen2 import greedy_generate_reference

MODEL_NAME = "tiny-qwen2"


@contextlib.contextmanager
def serve(tiny_model, tiny_tokenizer, engine_cls=LLMEngine, **cfg_kw):
    cfg = tiny_cfg(**cfg_kw)
    metrics = Metrics()
    engine = engine_cls(cfg, model=tiny_model, tokenizer=tiny_tokenizer, metrics=metrics)
    aeng = AsyncEngine(engine, max_waiting=cfg.max_waiting)
    app = create_app(aeng, tiny_tokenizer, MODEL_NAME, cfg.max_model_len, metrics)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}", aeng
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def server(tiny_model, tiny_tokenizer):
    with serve(tiny_model, tiny_tokenizer) as s:
        yield s


def _events(text: str) -> list[str]:
    assert text.endswith("\n\n")
    events = text.strip("\n").split("\n\n")
    assert all(e.startswith("data: ") for e in events), events[:3]
    return [e[len("data: "):] for e in events]


PROMPT = "The quick brown fox"


def _expected(tiny_model, tiny_tokenizer, prompt=PROMPT, n=12):
    ids = tiny_tokenizer.encode(prompt, add_special_tokens=False)
    toks = greedy_generate_reference(tiny_model, ids, n)
    return ids, toks, tiny_tokenizer.decode(toks, skip_special_tokens=True)


def test_health_and_models(server):
    url, _ = server
    assert httpx.get(f"{url}/health").json()["status"] == "ok"
    models = httpx.get(f"{url}/v1/models").json()
    assert models["data"][0]["id"] == MODEL_NAME


def test_completion_non_stream(server, tiny_model, tiny_tokenizer):
    url, _ = server
    ids, toks, text = _expected(tiny_model, tiny_tokenizer)
    r = httpx.post(f"{url}/v1/completions", json={"model": MODEL_NAME, "prompt": PROMPT, "max_tokens": 12,
                                                    "temperature": 0, "ignore_eos": True})
    body = r.json()
    assert r.status_code == 200
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == text
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": len(ids), "completion_tokens": 12, "total_tokens": len(ids) + 12}


def test_token_id_prompt(server, tiny_model, tiny_tokenizer):
    url, _ = server
    ids, toks, text = _expected(tiny_model, tiny_tokenizer)
    r = httpx.post(f"{url}/v1/completions", json={"prompt": ids, "max_tokens": 12, "temperature": 0,
                                                    "ignore_eos": True})
    assert r.json()["choices"][0]["text"] == text


def test_stream_sse_format(server, tiny_model, tiny_tokenizer):
    url, _ = server
    ids, toks, text = _expected(tiny_model, tiny_tokenizer)
    r = httpx.post(f"{url}/v1/completions", json={"prompt": PROMPT, "max_tokens": 12, "temperature": 0,
                                                    "ignore_eos": True, "stream": True,
                                                    "stream_options": {"include_usage": True}})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    usage_chunk = chunks[-1]
    assert usage_chunk["choices"] == [] and usage_chunk["usage"]["completion_tokens"] == 12
    body = chunks[:-1]
    assert "".join(c["choices"][0]["text"] for c in body) == text
    assert all(c["choices"][0]["finish_reason"] is None for c in body[:-1])
    assert body[-1]["choices"][0]["finish_reason"] == "length"
    assert body[-1]["usage"]["prompt_tokens"] == len(ids)
    assert len(body) > 1                                              # streamed, not one blob


def test_stop_string(server, tiny_model, tiny_tokenizer):
    url, _ = server
    ids, toks, text = _expected(tiny_model, tiny_tokenizer, n=20)
    stop = text[len(text) // 2:len(text) // 2 + 3]
    assert stop.strip(), "need a visible stop string"
    want = text[:text.index(stop)]
    for stream in (False, True):
        r = httpx.post(f"{url}/v1/completions", json={"prompt": PROMPT, "max_tokens": 20, "temperature": 0,
                                                        "ignore_eos": True, "stop": [stop], "stream": stream})
        if stream:
            chunks = [json.loads(e) for e in _events(r.text)[:-1]]
            got = "".join(c["choices"][0]["text"] for c in chunks)
            finish = chunks[-1]["choices"][0]["finish_reason"]
            used = chunks[-1]["usage"]["completion_tokens"]
        else:
            got, finish = r.json()["choices"][0]["text"], r.json()["choices"][0]["finish_reason"]
            used = r.json()["usage"]["completion_tokens"]
        assert got == want and finish == "stop"
        assert used < 20
        assert stop not in got


def test_chat_completion(server, tiny_tokenizer):
    url, _ = server
    msgs = [{"role": "user", "content": "hello"}]
    r = httpx.post(f"{url}/v1/chat/completions", json={"messages": msgs, "max_tokens": 8, "temperature": 0,
                                                         "ignore_eos": True})
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    prompt_ids = tiny_tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    prompt_ids = prompt_ids["input_ids"] if hasattr(prompt_ids, "keys") else prompt_ids
    assert body["usage"]["prompt_tokens"] == len(prompt_ids)
    r = httpx.post(f"{url}/v1/chat/completions", json={"messages": msgs, "max_tokens": 8, "temperature": 0,
                                                         "ignore_eos": True, "stream": True})
    chunks = [json.loads(e) for e in _events(r.text)[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == body["choices"][0]["message"]["content"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_openai_sdk_streams_unchanged(server):
    openai = pytest.importorskip("openai")
    url, _ = server
    client = openai.OpenAI(base_url=f"{url}/v1", api_key="unused")
    stream = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": "hi"}],
                                            max_tokens=10, temperature=0, stream=True,
                                            stream_options={"include_usage": True},
                                            extra_body={"ignore_eos": True})
    pieces, usage, finish = [], None, None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices:
            pieces.append(chunk.choices[0].delta.content or "")
            finish = chunk.choices[0].finish_reason or finish
    assert finish == "length" and usage.completion_tokens == 10
    comp = client.completions.create(model=MODEL_NAME, prompt="hello", max_tokens=5, temperature=0,
                                     extra_body={"ignore_eos": True})
    assert comp.usage.completion_tokens == 5


def test_rejects_too_long(server):
    url, _ = server
    r = httpx.post(f"{url}/v1/completions", json={"prompt": list(range(250)), "max_tokens": 10})
    assert r.status_code == 400 and "maximum context length" in r.json()["error"]["message"]


def test_metrics_endpoint(server):
    url, _ = server
    httpx.post(f"{url}/v1/completions", json={"prompt": PROMPT, "max_tokens": 3})
    text = httpx.get(f"{url}/metrics").text
    for name in ("num_requests_waiting", "num_requests_running", "kv_cache_usage_ratio", "num_preemptions_total",
                 "prompt_tokens_total", "generation_tokens_total", "time_to_first_token_seconds_bucket",
                 "time_per_output_token_seconds_bucket", "e2e_request_latency_seconds_bucket",
                 "step_duration_seconds_bucket", "batch_size_bucket"):
        assert name in text, name


def test_100_concurrent_streams(server, tiny_model, tiny_tokenizer):
    url, aeng = server
    prompts = [f"request number {i} says" for i in range(100)]

    async def one(client, i):
        r = await client.post(f"{url}/v1/completions", json={"prompt": prompts[i], "max_tokens": 16,
                                                               "temperature": 0, "ignore_eos": True,
                                                               "stream": True})
        chunks = [json.loads(e) for e in _events(r.text)[:-1]]
        return "".join(c["choices"][0]["text"] for c in chunks), chunks[-1]["usage"]["completion_tokens"]

    async def main():
        async with httpx.AsyncClient(timeout=120) as client:
            return await asyncio.gather(*(one(client, i) for i in range(100)))

    results = asyncio.run(main())
    for i in (0, 37, 99):                                        # greedy: same as alone
        ids = tiny_tokenizer.encode(prompts[i], add_special_tokens=False)
        want = tiny_tokenizer.decode(greedy_generate_reference(tiny_model, ids, 16), skip_special_tokens=True)
        assert results[i][0] == want
    assert all(n == 16 for _, n in results)
    eng = aeng.engine
    assert eng.stats["peak_running"] > 1
    assert eng.cache.num_free_blocks == eng.cache.num_blocks


def test_disconnect_frees_blocks(tiny_model, tiny_tokenizer):
    with serve(tiny_model, tiny_tokenizer, max_model_len=1024, kv_cache_mb=8) as (url, aeng):
        eng = aeng.engine
        with httpx.Client(timeout=30) as client:
            with client.stream("POST", f"{url}/v1/completions",
                               json={"prompt": PROMPT, "max_tokens": 1000, "temperature": 0, "ignore_eos": True,
                                     "stream": True}) as r:
                lines = r.iter_lines()
                for _ in range(6):
                    next(lines)
                assert eng.seqs                                      # running right now
        deadline = time.time() + 5                                   # connection closed: abort must land
        while (eng.seqs or eng.cache.num_free_blocks != eng.cache.num_blocks) and time.time() < deadline:
            time.sleep(0.01)
        assert not eng.seqs
        assert eng.cache.num_free_blocks == eng.cache.num_blocks
        metrics = httpx.get(f"{url}/metrics").text                 # gauges refreshed by the abort itself
        assert "num_requests_running 0.0" in metrics and "num_requests_waiting 0.0" in metrics
        assert "kv_cache_usage_ratio 0.0" in metrics


def test_503_when_queue_full(tiny_model, tiny_tokenizer):
    with serve(tiny_model, tiny_tokenizer, max_waiting=0) as (url, _):
        r = httpx.post(f"{url}/v1/completions", json={"prompt": PROMPT, "max_tokens": 4})
        assert r.status_code == 503 and r.headers["retry-after"] == "1"
        assert "request_rejected_total 1.0" in httpx.get(f"{url}/metrics").text


def test_overload_returns_503_not_crash(tiny_model, tiny_tokenizer):
    with serve(tiny_model, tiny_tokenizer, max_waiting=5, max_num_seqs=2) as (url, aeng):
        async def main():
            async with httpx.AsyncClient(timeout=120) as client:
                reqs = [client.post(f"{url}/v1/completions", json={"prompt": PROMPT, "max_tokens": 64,
                                                                     "ignore_eos": True}) for _ in range(40)]
                return await asyncio.gather(*reqs)

        codes = [r.status_code for r in asyncio.run(main())]
        assert set(codes) <= {200, 503} and 503 in codes and 200 in codes
        assert httpx.get(f"{url}/health").status_code == 200


def test_request_timeout(tiny_model, tiny_tokenizer):
    with serve(tiny_model, tiny_tokenizer, request_timeout=0.05, max_model_len=1024, kv_cache_mb=8) as (url, aeng):
        r = httpx.post(f"{url}/v1/completions", json={"prompt": PROMPT, "max_tokens": 900, "ignore_eos": True,
                                                        "stream": True})
        chunks = [json.loads(e) for e in _events(r.text)[:-1]]
        assert chunks[-1]["choices"][0]["finish_reason"] == "timeout"
        r = httpx.post(f"{url}/v1/completions", json={"prompt": PROMPT, "max_tokens": 900, "ignore_eos": True})
        assert r.status_code == 504
        eng = aeng.engine
        assert eng.cache.num_free_blocks == eng.cache.num_blocks


def test_static_engine_serves_same_api(tiny_model, tiny_tokenizer):
    """Baseline B behind the same server: batches of 4 (or 30 ms), same greedy outputs."""
    with serve(tiny_model, tiny_tokenizer, engine_cls=StaticBatchEngine, static_batch_size=4,
               static_batch_wait_ms=30) as (url, aeng):
        prompts = [f"static request {i}" for i in range(10)]

        async def main():
            async with httpx.AsyncClient(timeout=60) as client:
                reqs = [client.post(f"{url}/v1/completions", json={"prompt": p, "max_tokens": 6 + i,
                                                                     "temperature": 0, "ignore_eos": True,
                                                                     "stream": True})
                        for i, p in enumerate(prompts)]
                return await asyncio.gather(*reqs)

        for i, r in enumerate(asyncio.run(main())):
            chunks = [json.loads(e) for e in _events(r.text)[:-1]]
            ids = tiny_tokenizer.encode(prompts[i], add_special_tokens=False)
            want = tiny_tokenizer.decode(greedy_generate_reference(tiny_model, ids, 6 + i), skip_special_tokens=True)
            assert "".join(c["choices"][0]["text"] for c in chunks) == want
            assert chunks[-1]["usage"]["completion_tokens"] == 6 + i
        assert aeng.engine.stats["batches"] >= 3 and 0 < aeng.engine.waste < 1
