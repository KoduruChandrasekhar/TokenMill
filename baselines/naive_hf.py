"""Baseline A (Part 1): HuggingFace generate(), one request at a time.

    python -m baselines.naive_hf --port 8001          # fp16 on CUDA, fp32 on CPU

POST /generate is the plan's minimal endpoint (chat template applied, {prompt, max_tokens,
temperature, stream}). POST /v1/completions speaks the same OpenAI streaming format as the real
server, so the benchmark client drives both unchanged.

One request at a time: an asyncio.Lock is held until generate() has really stopped. generate()
runs in a worker thread so the event loop stays responsive. When a client disconnects, the
finally block sets a stop event (a StoppingCriteria checks it every step) and waits for the worker
inside a shielded cancel scope; Starlette cancels a disconnected stream through anyio, which would
otherwise interrupt that wait and release the lock while the GPU is still busy.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import queue
import threading
import time

import anyio
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from infer.api.protocol import CompletionRequest, new_id, now, usage
from infer.detokenizer import IncrementalDetokenizer
from infer.engine.sequence import SamplingParams


class StopOnEvent(StoppingCriteria):                 # lets the endpoint stop generate() early
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


class TokenStreamer(BaseStreamer):
    """Pushes each new token id into a queue the moment generate() produces it. (HF's
    TextIteratorStreamer buffers text up to word boundaries, which would distort time to first
    token; detokenization happens on the consumer side with the shared incremental detokenizer.)"""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.seen_prompt = False

    def put(self, value):
        if not self.seen_prompt:                     # the first call carries the prompt
            self.seen_prompt = True
            return
        for t in value.reshape(-1).tolist():
            self.q.put(t)

    def end(self):
        self.q.put(None)

    def __iter__(self):
        while (tok := self.q.get()) is not None:
            yield tok


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


def build_app(model_name: str, dtype: str, device: str, max_model_len: int) -> FastAPI:
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dt = {"auto": torch.float16 if dev.type == "cuda" else torch.float32, "float16": torch.float16,
          "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt).to(dev).eval()
    gpu_lock = asyncio.Lock()
    app = FastAPI(title="naive-hf")

    def run_generate(prompt_ids: list[int], params: SamplingParams, streamer=None, criterion=None) -> list[int]:
        x = torch.tensor([prompt_ids], device=dev)
        kwargs = dict(max_new_tokens=params.max_tokens, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
        if params.greedy:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            kwargs.update(do_sample=True, temperature=params.temperature, top_p=params.top_p,
                          top_k=params.top_k if params.top_k > 0 else None)
            if params.seed is not None:
                torch.manual_seed(params.seed)
        if params.ignore_eos:
            kwargs["min_new_tokens"] = params.max_tokens        # forces exact output length
        with torch.inference_mode():
            out = model.generate(x, attention_mask=torch.ones_like(x), streamer=streamer,
                                 stopping_criteria=StoppingCriteriaList([criterion]) if criterion else None,
                                 **kwargs)
        return out[0, len(prompt_ids):].tolist()

    async def stream_tokens(prompt_ids: list[int], params: SamplingParams):
        """Yields token ids as generate() produces them; holds the lock until generate() has stopped."""
        async with gpu_lock:
            stop = threading.Event()
            streamer = TokenStreamer()
            worker = threading.Thread(target=run_generate, args=(prompt_ids, params, streamer, StopOnEvent(stop)))
            worker.start()
            try:
                it = iter(streamer)
                while (tok := await asyncio.to_thread(next, it, None)) is not None:
                    yield tok
            finally:                                    # also runs when the client disconnects
                stop.set()                              # generate() exits at its next step
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(worker.join)

    def chat_ids(prompt: str) -> list[int]:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                            tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "naive-hf"}]}

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature)
        ids = chat_ids(req.prompt)
        if not req.stream:
            async with gpu_lock:                        # one request at a time
                out = await asyncio.to_thread(run_generate, ids, params)
            return {"text": tokenizer.decode(out, skip_special_tokens=True)}

        async def events():
            detok = IncrementalDetokenizer(tokenizer)
            async for tok in stream_tokens(ids, params):
                if piece := detok.add([tok]):
                    yield f"data: {json.dumps({'text': piece})}\n\n"
            if piece := detok.flush():
                yield f"data: {json.dumps({'text': piece})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt = req.prompt
        if isinstance(prompt, list) and prompt and not isinstance(prompt[0], int):
            prompt = prompt[0]
        ids = list(prompt) if isinstance(prompt, list) else tokenizer.encode(prompt, add_special_tokens=False)
        params = req.sampling_params(default_max_tokens=16)
        if len(ids) + params.max_tokens > max_model_len:
            return JSONResponse({"error": {"message": f"maximum context length is {max_model_len}"}}, 400)
        rid, served = new_id("cmpl"), req.model or model_name

        async def run():
            detok = IncrementalDetokenizer(tokenizer, params.stop)
            n = 0
            eos = set(model.generation_config.eos_token_id or [])
            # aclosing: leaving the loop early (stop string) must stop generate() and release the lock now
            async with contextlib.aclosing(stream_tokens(ids, params)) as tokens:
                async for tok in tokens:
                    n += 1
                    text = detok.add([tok])
                    if detok.stopped:
                        yield text, "stop", detok.num_tokens_at_stop
                        return
                    if text:
                        yield text, None, n
            reason = "stop" if (not params.ignore_eos and n < params.max_tokens and detok.ids
                                and detok.ids[-1] in eos) else "length"
            yield detok.flush(), reason, n

        if not req.stream:
            parts, reason, n = [], None, 0
            async for text, reason, n in run():
                parts.append(text)
            return {"id": rid, "object": "text_completion", "created": now(), "model": served,
                    "choices": [{"index": 0, "text": "".join(parts), "logprobs": None, "finish_reason": reason}],
                    "usage": usage(len(ids), n)}

        async def events():
            created = now()
            n = 0
            async for text, reason, n in run():
                chunk = {"id": rid, "object": "text_completion", "created": created, "model": served,
                         "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": reason}]}
                if reason:
                    chunk["usage"] = usage(len(ids), n)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            if req.include_usage:
                yield f"data: {json.dumps({'id': rid, 'object': 'text_completion', 'created': created, 'model': served, 'choices': [], 'usage': usage(len(ids), n)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="baseline A: HF generate, one request at a time")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    t0 = time.time()
    app = build_app(args.model, args.dtype, args.device, args.max_model_len)
    print(f"model loaded in {time.time() - t0:.1f}s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
