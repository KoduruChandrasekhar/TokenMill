"""OpenAI-compatible request/response models (the subset clients and benchmark tools use).

Unknown fields are accepted and ignored, so clients that send extra OpenAI parameters still work.
ignore_eos and top_k are vLLM-style extensions that benchmarks rely on.
"""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, ConfigDict, Field

from infer.engine.sequence import SamplingParams


class _Request(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: dict | None = None
    n: int = 1
    ignore_eos: bool = False

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.get("include_usage"))

    def sampling_params(self, default_max_tokens: int) -> SamplingParams:
        stop = [self.stop] if isinstance(self.stop, str) else list(self.stop or [])
        max_tokens = self.max_completion_tokens or self.max_tokens or default_max_tokens
        return SamplingParams(max_tokens=max_tokens, temperature=self.temperature, top_p=self.top_p,
                              top_k=self.top_k, stop=stop, seed=self.seed, ignore_eos=self.ignore_eos)


class CompletionRequest(_Request):
    prompt: str | list[int] | list[str] | list[list[int]]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list | None = None


class ChatCompletionRequest(_Request):
    messages: list[ChatMessage]


class ErrorResponse(BaseModel):
    error: dict


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def usage(prompt: int, completion: int) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def now() -> int:
    return int(time.time())


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=now)
    owned_by: str = "infer"
    max_model_len: int | None = None
