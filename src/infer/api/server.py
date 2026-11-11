"""FastAPI app: OpenAI-compatible routes, SSE streaming, backpressure, metrics.

The API side lives on the asyncio event loop and never touches the GPU. A request is tokenized,
handed to the engine thread through AsyncEngine's inbox, and its tokens come back through an
asyncio.Queue. Detokenization and stop strings happen here, on the way out. When a client
disconnects mid-stream, Starlette cancels the generator; the finally block sends an abort through
the same inbox and the engine frees the sequence's blocks at its next step.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from infer.api.protocol import ChatCompletionRequest, CompletionRequest, ModelCard, new_id, now, usage
from infer.detokenizer import IncrementalDetokenizer
from infer.engine.engine import AsyncEngine, QueueFull
from infer.engine.sequence import SamplingParams
from infer.metrics import Metrics

log = logging.getLogger("infer.api")


def _error(status: int, message: str, kind: str = "invalid_request_error", headers: dict | None = None):
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status_code=status,
                        headers=headers)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def create_app(engine: AsyncEngine, tokenizer, served_model_name: str, max_model_len: int,
               metrics: Metrics | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        yield
        engine.shutdown()

    app = FastAPI(title="infer", lifespan=lifespan)
    app.state.engine = engine

    # --- plain routes -----------------------------------------------------------------------------

    @app.get("/health")
    async def health():
        if not engine.is_running:
            return _error(503, "engine thread is not running", "server_error")
        return {"status": "ok", "waiting": engine.num_waiting, "last_error": engine.last_error}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [ModelCard(id=served_model_name, max_model_len=max_model_len).model_dump()]}

    @app.get("/metrics")
    async def prometheus():
        if metrics is None:
            return Response("", media_type=CONTENT_TYPE_LATEST)
        return Response(metrics.render(), media_type=CONTENT_TYPE_LATEST)

    # --- generation -------------------------------------------------------------------------------

    def _check_lengths(prompt_ids: list[int], params: SamplingParams):
        if not prompt_ids:
            return _error(400, "prompt is empty")
        if len(prompt_ids) + params.max_tokens > max_model_len:
            return _error(400, f"This model's maximum context length is {max_model_len} tokens. However, you "
                               f"requested {len(prompt_ids) + params.max_tokens} tokens ({len(prompt_ids)} in "
                               f"the prompt, {params.max_tokens} for the completion).")
        if params.max_tokens < 1:
            return _error(400, "max_tokens must be at least 1")
        if params.temperature < 0 or not 0 < params.top_p <= 1:
            return _error(400, "temperature must be >= 0 and top_p in (0, 1]")
        return None

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        prompt = req.prompt
        if isinstance(prompt, list) and prompt and not isinstance(prompt[0], int):
            if len(prompt) != 1:
                return _error(400, "batched prompts are not supported; send one prompt per request")
            prompt = prompt[0]
        prompt_ids = prompt if isinstance(prompt, list) else tokenizer.encode(prompt, add_special_tokens=False)
        if req.n != 1:
            return _error(400, "only n=1 is supported")
        params = req.sampling_params(default_max_tokens=16)
        if (err := _check_lengths(prompt_ids, params)) is not None:
            return err
        return await _generate(request, "completion", req, prompt_ids, params)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        try:
            prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        except Exception as e:  # malformed messages or template error
            return _error(400, f"could not apply chat template: {e}")
        if hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        prompt_ids = list(prompt_ids)
        if req.n != 1:
            return _error(400, "only n=1 is supported")
        params = req.sampling_params(default_max_tokens=max(1, max_model_len - len(prompt_ids)))
        if (err := _check_lengths(prompt_ids, params)) is not None:
            return err
        return await _generate(request, "chat", req, prompt_ids, params)

    async def _generate(request: Request, kind: str, req, prompt_ids: list[int], params: SamplingParams):
        rid = new_id("chatcmpl" if kind == "chat" else "cmpl")
        try:
            q = engine.add_request(rid, prompt_ids, params)
        except QueueFull as e:
            if metrics:
                metrics.request_rejected_total.inc()
            return _error(503, f"server overloaded: {e}; retry later", "overloaded", headers={"Retry-After": "1"})
        model = req.model or served_model_name
        if req.stream:
            return StreamingResponse(_stream(rid, kind, model, q, prompt_ids, params, req.include_usage),
                                     media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return await _collect(rid, kind, model, q, prompt_ids, params, request)

    async def _consume(rid, q: asyncio.Queue, params: SamplingParams):
        """Yield (text_delta, finish_reason, completion_tokens) as engine outputs arrive."""
        detok = IncrementalDetokenizer(tokenizer, params.stop)
        n_tokens = 0
        while True:
            out = await q.get()
            if out.error:
                raise RuntimeError(out.error)
            delta = detok.add(out.new_token_ids)
            if detok.stopped:
                n_tokens = detok.num_tokens_at_stop      # tokens up to the one completing the stop string
                if not out.finished:
                    engine.abort(rid)                 # stop string: free the sequence now
                yield delta, "stop", n_tokens
                return
            n_tokens += len(out.new_token_ids)
            if out.finished:
                yield delta + detok.flush(), out.finish_reason, n_tokens
                return
            yield delta, None, n_tokens

    def _chunk(rid, kind, model, created, text, finish_reason, role=False):
        if kind == "chat":
            delta = {"role": "assistant", "content": text} if role else ({"content": text} if text else {})
            choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
            obj = "chat.completion.chunk"
        else:
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}
            obj = "text_completion"
        return {"id": rid, "object": obj, "created": created, "model": model, "choices": [choice]}

    async def _stream(rid, kind, model, q, prompt_ids, params, include_usage):
        created = now()
        done = False
        n_tokens = 0
        try:
            if kind == "chat":
                yield _sse(_chunk(rid, kind, model, created, "", None, role=True))
            async for text, finish_reason, n_tokens in _consume(rid, q, params):
                if text or finish_reason:
                    chunk = _chunk(rid, kind, model, created, text, finish_reason)
                    if finish_reason:
                        chunk["usage"] = usage(len(prompt_ids), n_tokens)
                    yield _sse(chunk)
            if include_usage:
                yield _sse({"id": rid, "object": "chat.completion.chunk" if kind == "chat" else "text_completion",
                            "created": created, "model": model, "choices": [],
                            "usage": usage(len(prompt_ids), n_tokens)})
            yield "data: [DONE]\n\n"
            done = True
        except RuntimeError as e:
            yield _sse({"error": {"message": str(e), "type": "server_error"}})
            yield "data: [DONE]\n\n"
            done = True
        finally:
            if not done:                     # client went away (CancelledError / GeneratorExit)
                engine.abort(rid)

    async def _collect(rid, kind, model, q, prompt_ids, params, request: Request):
        parts, finish_reason, n_tokens = [], None, 0
        try:
            async for text, finish_reason, n_tokens in _consume(rid, q, params):
                parts.append(text)
        except RuntimeError as e:
            return _error(500, str(e), "server_error")
        except asyncio.CancelledError:
            engine.abort(rid)
            raise
        if finish_reason == "timeout":
            return _error(504, "request timed out", "timeout")
        text = "".join(parts)
        if kind == "chat":
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": None,
                      "finish_reason": finish_reason}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}
            obj = "text_completion"
        return {"id": rid, "object": obj, "created": now(), "model": model, "choices": [choice],
                "usage": usage(len(prompt_ids), n_tokens)}

    return app
