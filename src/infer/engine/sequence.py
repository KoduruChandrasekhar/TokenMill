"""Sequence, SamplingParams and the status enum shared by the scheduler, engine and API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(Enum):
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2      # blocks released; resumes by recomputing prompt + generated tokens
    FINISHED = 3
    SWAPPED = 4        # blocks copied to CPU; resumes by copying them back


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] = field(default_factory=list)   # checked by the API layer on decoded text
    seed: int | None = None
    ignore_eos: bool = False                          # needed for fair benchmarks

    @property
    def greedy(self) -> bool:
        return self.temperature <= 1e-5 or self.top_k == 1


@dataclass(eq=False)          # identity semantics: two requests with equal fields are still two requests
class Sequence:
    id: str
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    slot: int | None = None                            # contiguous cache slot (Part 4)
    block_table: list[int] = field(default_factory=list)   # paged cache blocks (Part 5)
    arrival_time: float = 0.0
    first_token_time: float | None = None
    finish_time: float | None = None
    finish_reason: str | None = None                   # stop | length | abort | timeout
    out_queue: asyncio.Queue | None = None             # where the engine thread sends outputs
    # engine bookkeeping
    num_computed_tokens: int = 0                       # tokens whose K/V are in the cache
    admit_order: int = -1                              # increases on every (re)admission; LIFO preemption
    num_preemptions: int = 0
    cpu_block_table: list[int] = field(default_factory=list)   # while SWAPPED
    block_hashes: list[int] = field(default_factory=list)      # prefix caching
    num_registered_blocks: int = 0
    generator: Any = None                              # torch.Generator for seeded sampling

    def __repr__(self) -> str:
        return (f"Sequence({self.id!r}, {self.status.name}, prompt={self.num_prompt_tokens}, "
                f"out={self.num_output_tokens}, computed={self.num_computed_tokens}, blocks={self.block_table})")

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def last_token(self) -> int:
        return self.output_ids[-1] if self.output_ids else self.prompt_ids[-1]

    @property
    def num_new_tokens(self) -> int:
        """Tokens this sequence feeds to the next forward pass: the whole uncached tail."""
        return self.length - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == Status.FINISHED


@dataclass
class RequestOutput:
    """One engine step's result for one request; the API turns these into SSE chunks."""

    request_id: str
    new_token_ids: list[int]
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_output_tokens: int
    arrival_time: float = 0.0
    first_token_time: float | None = None
    finish_time: float | None = None
    num_preemptions: int = 0
    error: str | None = None
