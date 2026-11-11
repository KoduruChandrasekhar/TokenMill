"""The continuous-batching engine and the thread that runs it.

LLMEngine is synchronous: add_request / abort_request / step(). Every step re-plans the batch, so
new requests join at the next step and finished ones leave immediately. Tests and offline
experiments drive it directly.

AsyncEngine runs an engine on its own thread, because GPU calls block. The API puts new requests
and aborts into a thread-safe inbox and sets a wake event; each request owns an asyncio.Queue that
the engine thread fills with loop.call_soon_threadsafe, never touching asyncio objects directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import traceback

import torch

from infer.config import EngineConfig
from infer.engine.model_runner import ModelRunner
from infer.engine.scheduler import Scheduler
from infer.engine.sequence import RequestOutput, SamplingParams, Sequence, Status
from infer.kv.contiguous import ContiguousKVPool, SlotManager
from infer.kv.paged_cache import BlockManager, KVPool
from infer.metrics import Metrics
from infer.model.loader import load_model, load_tokenizer, resolve_device, resolve_dtype, resolve_model_path
from infer.sampling import make_generator

log = logging.getLogger("infer.engine")
req_log = logging.getLogger("infer.requests")

CPU_DEFAULT_KV_MB = 256


class RequestRejected(ValueError):
    """The request can never be served (for example, its prompt is longer than max_model_len)."""


def kv_budget_bytes(cfg: EngineConfig, runner: ModelRunner, device: torch.device) -> int:
    """KV pool size: the --kv-cache-mb cap if given, else what is left of
    GPU memory x utilization after weights and peak activations."""
    if cfg.kv_cache_mb is not None:
        return int(cfg.kv_cache_mb * 2**20)
    if device.type != "cuda":
        return CPU_DEFAULT_KV_MB * 2**20
    peak = runner.profile_peak_memory(cfg.max_num_batched_tokens, cfg.max_num_seqs)
    total = torch.cuda.get_device_properties(device).total_memory
    # Graphs are captured after the pool exists (they bake in its address), so their memory
    # must be set aside first: 128 graphs did not fit next to a 90% pool on a T4.
    graphs = int(cfg.cuda_graph_reserve_mb * 2**20) if cfg.cuda_graphs else 0
    budget = int(total * cfg.gpu_memory_utilization) - peak - graphs
    log.info("memory profile: total %.0f MiB, peak (weights + activations) %.0f MiB, graphs %.0f MiB, "
             "KV budget %.0f MiB", total / 2**20, peak / 2**20, graphs / 2**20, budget / 2**20)
    if budget <= 0:
        raise RuntimeError("no memory left for the KV cache; lower --max-batched-tokens or raise utilization")
    return budget


def build_cache(cfg: EngineConfig, model, runner: ModelRunner, dtype, device):
    bytes_per_token = model.cfg.kv_bytes_per_token(torch.tensor([], dtype=dtype).element_size())
    budget = kv_budget_bytes(cfg, runner, device)
    if cfg.cache_type == "contiguous":
        num_slots = min(cfg.max_num_seqs, budget // (bytes_per_token * cfg.max_model_len))
        if num_slots < 1:
            raise RuntimeError("KV budget too small for even one max_model_len slot")
        pool = ContiguousKVPool(model.cfg, num_slots, cfg.max_model_len, dtype, device)
        return SlotManager(pool)
    num_blocks = budget // (bytes_per_token * cfg.block_size)
    if num_blocks < 2:
        raise RuntimeError("KV budget too small for the paged pool")
    reserve = 1 if cfg.cuda_graphs else 0        # one scratch block that padded graph rows write into
    pool = KVPool(model.cfg, num_blocks + reserve, cfg.block_size, dtype, device)
    swap_pool = None
    if cfg.preemption_mode == "swap":
        cpu_blocks = int(cfg.swap_space_mb * 2**20) // (bytes_per_token * cfg.block_size)
        swap_pool = KVPool(model.cfg, max(cpu_blocks, 1), cfg.block_size, dtype, torch.device("cpu"))
        if device.type == "cuda":
            swap_pool.kv = swap_pool.kv.pin_memory()
    return BlockManager(pool, cfg.watermark, cfg.enable_prefix_caching, swap_pool, num_usable_blocks=num_blocks)


class LLMEngine:
    def __init__(self, cfg: EngineConfig, model=None, tokenizer=None, metrics: Metrics | None = None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device) if model is None else model.embed_tokens.weight.device
        self.dtype = resolve_dtype(cfg.dtype, self.device) if model is None else model.embed_tokens.weight.dtype
        if model is None:
            path = resolve_model_path(cfg.model)
            model = load_model(path, self.dtype, self.device, max_positions=cfg.max_model_len)
            tokenizer = tokenizer or load_tokenizer(path)
        self.model = model
        self.dtype = model.embed_tokens.weight.dtype     # the KV cache always matches the weights
        self.tokenizer = tokenizer
        self.eos_ids = set(model.cfg.eos_token_ids)
        self.runner = ModelRunner(model, None, self.device, cfg.max_model_len)
        self.cache = build_cache(cfg, model, self.runner, self.dtype, self.device)
        self.runner.cache = self.cache
        self.scheduler = Scheduler(cfg, self.cache)
        self.metrics = metrics
        self.seqs: dict[str, Sequence] = {}
        self._rng = torch.Generator().manual_seed(cfg.seed)
        self.stats = {"steps": 0, "peak_running": 0, "prefill_steps": 0, "decode_steps": 0,
                      "generated_tokens": 0, "busy_time": 0.0}
        if cfg.cuda_graphs:
            if self.device.type != "cuda" or cfg.cache_type != "paged":
                log.warning("--cuda-graphs needs CUDA and the paged cache; running without graphs")
            else:
                from infer.engine.cuda_graphs import CudaGraphRunner

                try:
                    graphs = CudaGraphRunner(self.model, self.cache, cfg)
                    graphs.capture()
                    self.runner.graphs = graphs
                except Exception:                    # e.g. a kernel that is not capture-safe
                    log.exception("CUDA graph capture failed; decoding eagerly")
        log.info("engine ready: %s cache, %d blocks of %d tokens (%.0f MiB), dtype %s, device %s",
                 cfg.cache_type, self.cache.num_blocks, self.cache.block_size,
                 self.cache.pool.nbytes / 2**20, self.dtype, self.device)

    # --- requests ---------------------------------------------------------------------------------

    def add_request(self, request_id: str, prompt_ids: list[int], params: SamplingParams,
                    arrival_time: float | None = None) -> Sequence:
        if not prompt_ids:
            raise RequestRejected("prompt is empty")
        if len(prompt_ids) >= self.cfg.max_model_len:
            raise RequestRejected(f"prompt has {len(prompt_ids)} tokens; max_model_len is {self.cfg.max_model_len}")
        if request_id in self.seqs:
            raise RequestRejected(f"duplicate request id {request_id}")
        seq = Sequence(request_id, list(prompt_ids), params,
                       arrival_time=time.monotonic() if arrival_time is None else arrival_time)
        if not params.greedy:
            seq.generator = make_generator(params.seed, self.device, self._rng)
        self.seqs[request_id] = seq
        self.scheduler.add(seq)
        return seq

    def abort_request(self, request_id: str, reason: str = "abort") -> RequestOutput | None:
        seq = self.seqs.pop(request_id, None)
        if seq is None:
            return None
        self.scheduler.finish(seq, reason)
        seq.finish_time = time.monotonic()
        self._on_finished(seq)
        self._update_gauges()                  # an abort on an idle engine must not leave gauges stale
        return self._output(seq, [])

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    has_work = has_unfinished

    def idle_timeout(self) -> float | None:
        return None

    @property
    def num_waiting(self) -> int:
        return len(self.scheduler.waiting)

    # --- one step ---------------------------------------------------------------------------------

    def step(self) -> list[RequestOutput]:
        outputs: list[RequestOutput] = []
        if self.cfg.request_timeout:
            self._expire(outputs)
        t0 = time.perf_counter()
        plan = self.scheduler.schedule()
        for seq in plan.ignored:                        # can never fit in the KV pool
            self.seqs.pop(seq.id, None)
            self.scheduler.finish(seq, "length")
            seq.finish_time = time.monotonic()
            self._on_finished(seq)
            outputs.append(self._output(seq, [], error=None if seq.output_ids else
                                        "prompt does not fit in the KV cache"))
        if plan.empty:
            return outputs
        n_prefill_tokens = plan.num_prefill_tokens
        sampled, tokens = self.runner.execute(plan.prefill_seqs, plan.decode_seqs, plan.prefill_chunks)
        now = time.monotonic()
        step_time = time.perf_counter() - t0
        for seq in plan.prefill_seqs:                   # a chunk may cover only part of the prompt
            seq.num_computed_tokens += plan.chunk(seq)
            self.cache.register_computed(seq)
        for seq in plan.decode_seqs:
            seq.num_computed_tokens = seq.length
            self.cache.register_computed(seq)
        for seq, tok in zip(sampled, tokens):
            seq.output_ids.append(tok)
            if seq.first_token_time is None:
                seq.first_token_time = now
            reason = self._stop_reason(seq, tok)
            if reason:
                self.scheduler.finish(seq, reason)
                seq.finish_time = now
                self.seqs.pop(seq.id, None)
                self._on_finished(seq)
            outputs.append(self._output(seq, [tok]))
        self._record_step(plan, n_prefill_tokens, step_time, len(sampled))
        return outputs

    def _stop_reason(self, seq: Sequence, tok: int) -> str | None:
        if tok in self.eos_ids and not seq.params.ignore_eos:
            return "stop"
        if seq.num_output_tokens >= seq.params.max_tokens or seq.length >= self.cfg.max_model_len:
            return "length"
        return None

    def _expire(self, outputs: list[RequestOutput]) -> None:
        deadline = time.monotonic() - self.cfg.request_timeout
        for seq in [s for s in self.seqs.values() if s.arrival_time < deadline]:
            outputs.append(self.abort_request(seq.id, reason="timeout"))

    def _output(self, seq: Sequence, new_tokens: list[int], error: str | None = None) -> RequestOutput:
        return RequestOutput(seq.id, new_tokens, seq.is_finished, seq.finish_reason, seq.num_prompt_tokens,
                             seq.num_output_tokens, seq.arrival_time, seq.first_token_time, seq.finish_time,
                             seq.num_preemptions, error)

    def _on_finished(self, seq: Sequence) -> None:
        if self.metrics:
            self.metrics.observe_finished(seq)
        if req_log.isEnabledFor(logging.INFO):
            ttft = (seq.first_token_time - seq.arrival_time) if seq.first_token_time else None
            req_log.info(json.dumps({
                "id": seq.id, "finish_reason": seq.finish_reason, "prompt_tokens": seq.num_prompt_tokens,
                "output_tokens": seq.num_output_tokens, "ttft_s": ttft,
                "e2e_s": (seq.finish_time - seq.arrival_time) if seq.finish_time else None,
                "preemptions": seq.num_preemptions}))

    def _record_step(self, plan, n_prefill_tokens: int, step_time: float, n_generated: int) -> None:
        s = self.stats
        s["steps"] += 1
        s["busy_time"] += step_time
        s["prefill_steps" if plan.prefill_seqs else "decode_steps"] += 1
        s["generated_tokens"] += n_generated
        running = len(self.scheduler.running)
        s["peak_running"] = max(s["peak_running"], running)
        m = self.metrics
        if m:
            m.step_duration_seconds.observe(step_time)
            m.batch_size.observe(len(plan.seqs))
            self._update_gauges()
            # first prefill only (a sequence resumed by recompute already has > 1 output token)
            m.prompt_tokens_total.inc(sum(s.num_prompt_tokens for s in plan.prefill_seqs if s.num_output_tokens == 1))
            m.generation_tokens_total.inc(n_generated)
            if plan.preempted:
                m.num_preemptions_total.inc(len(plan.preempted))
        if self.cfg.debug_steps:
            log.info("step=%d batch=%d prefill_tokens=%d decode_tokens=%d step_ms=%.2f kv_usage=%.3f "
                     "waiting=%d running=%d preempted=%d", s["steps"], len(plan.seqs), n_prefill_tokens,
                     len(plan.decode_seqs), step_time * 1e3, self.cache.usage(), len(self.scheduler.waiting),
                     running, len(plan.preempted))

    def _update_gauges(self) -> None:
        m = self.metrics
        if not m:
            return
        m.num_requests_waiting.set(len(self.scheduler.waiting))
        m.num_requests_running.set(len(self.scheduler.running))
        m.num_requests_swapped.set(sum(1 for q in self.scheduler.waiting if q.status == Status.SWAPPED))
        m.kv_cache_usage_ratio.set(self.cache.usage())
        m.kv_free_blocks.set(self.cache.num_free_blocks)
        if self.device.type == "cuda":
            m.gpu_memory_allocated_bytes.set(torch.cuda.memory_allocated(self.device))

    # --- offline helper -----------------------------------------------------------------------------

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]) -> list[list[int]]:
        """Run every prompt to completion through the step loop; returns output token ids in order."""
        plist = params if isinstance(params, list) else [params] * len(prompts)
        seqs = []
        for p, sp in zip(prompts, plist):
            self._offline_counter = getattr(self, "_offline_counter", 0) + 1
            seqs.append(self.add_request(f"offline-{self._offline_counter}", p, sp))
        while self.has_unfinished():
            self.step()
        return [s.output_ids for s in seqs]


class QueueFull(RuntimeError):
    pass


class AsyncEngine:
    """Runs an engine (continuous or static) on a dedicated thread."""

    def __init__(self, engine, max_waiting: int = 1000):
        self.engine = engine
        self.max_waiting = max_waiting
        self.inbox: queue.Queue = queue.Queue()
        self.wake = threading.Event()
        self.queues: dict[str, asyncio.Queue] = {}       # touched only by the engine thread
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self._stop = False
        self.last_error: str | None = None

    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self.thread.start()

    def shutdown(self) -> None:
        self._stop = True
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=30)

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def num_waiting(self) -> int:
        return self.inbox.qsize() + self.engine.num_waiting

    def add_request(self, request_id: str, prompt_ids: list[int], params: SamplingParams) -> asyncio.Queue:
        if self.num_waiting >= self.max_waiting:
            raise QueueFull(f"{self.num_waiting} requests already waiting")
        q: asyncio.Queue = asyncio.Queue()
        self.inbox.put(("add", request_id, prompt_ids, params, q, time.monotonic()))
        self.wake.set()
        return q

    def abort(self, request_id: str) -> None:
        self.inbox.put(("abort", request_id))
        self.wake.set()

    # --- engine thread ------------------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop:
            self.wake.clear()
            self._drain_inbox()
            if not self.engine.has_work():
                self.wake.wait(timeout=self.engine.idle_timeout())
                continue
            try:
                outputs = self.engine.step()
            except Exception as e:                          # keep serving; fail only what was in flight
                self.last_error = f"{type(e).__name__}: {e}"
                log.error("engine step failed:\n%s", traceback.format_exc())
                outputs = [o for rid in list(self.engine.seqs) if (o := self.engine.abort_request(rid, "error"))]
                for o in outputs:
                    o.error = self.last_error
            for out in outputs:
                self._emit(out)

    def _drain_inbox(self) -> None:
        while True:
            try:
                msg = self.inbox.get_nowait()
            except queue.Empty:
                return
            if msg[0] == "add":
                _, rid, prompt_ids, params, q, arrival = msg
                self.queues[rid] = q
                try:
                    self.engine.add_request(rid, prompt_ids, params, arrival_time=arrival)
                except RequestRejected as e:
                    self._emit(RequestOutput(rid, [], True, "error", len(prompt_ids), 0, error=str(e)))
            else:
                out = self.engine.abort_request(msg[1])
                if out is not None:
                    self._emit(out)

    def _emit(self, out: RequestOutput) -> None:
        q = self.queues.pop(out.request_id, None) if out.finished else self.queues.get(out.request_id)
        if q is None or self.loop is None or self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(q.put_nowait, out)
