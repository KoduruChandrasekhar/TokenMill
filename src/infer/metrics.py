"""Prometheus metrics for /metrics. Each engine gets its own registry so tests can build many."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
                    10.0, 20.0, 30.0, 60.0, 120.0)
_STEP_BUCKETS = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2,
                 0.5, 1.0, 2.5, 5.0)
_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        r = self.registry = registry or CollectorRegistry()
        self.num_requests_waiting = Gauge("num_requests_waiting", "Requests waiting to be admitted", registry=r)
        self.num_requests_running = Gauge("num_requests_running", "Requests in the running batch", registry=r)
        self.num_requests_swapped = Gauge("num_requests_swapped", "Requests swapped to CPU", registry=r)
        self.kv_cache_usage_ratio = Gauge("kv_cache_usage_ratio", "Fraction of KV blocks in use", registry=r)
        self.kv_free_blocks = Gauge("kv_free_blocks", "Free KV blocks (or slots)", registry=r)
        self.gpu_memory_allocated_bytes = Gauge("gpu_memory_allocated_bytes",
                                                "torch.cuda.memory_allocated()", registry=r)
        self.num_preemptions_total = Counter("num_preemptions", "Sequences preempted", registry=r)
        self.prompt_tokens_total = Counter("prompt_tokens", "Prompt tokens prefilled", registry=r)
        self.generation_tokens_total = Counter("generation_tokens", "Tokens generated", registry=r)
        self.request_success_total = Counter("request_success", "Finished requests", ["finish_reason"],
                                             registry=r)
        self.request_rejected_total = Counter("request_rejected", "Requests refused with 503", registry=r)
        self.time_to_first_token_seconds = Histogram("time_to_first_token_seconds", "TTFT",
                                                     buckets=_LATENCY_BUCKETS, registry=r)
        self.time_per_output_token_seconds = Histogram("time_per_output_token_seconds", "TPOT",
                                                       buckets=_STEP_BUCKETS, registry=r)
        self.e2e_request_latency_seconds = Histogram("e2e_request_latency_seconds", "Arrival to last token",
                                                     buckets=_LATENCY_BUCKETS, registry=r)
        self.step_duration_seconds = Histogram("step_duration_seconds", "One engine step",
                                               buckets=_STEP_BUCKETS, registry=r)
        self.batch_size = Histogram("batch_size", "Sequences per step", buckets=_BATCH_BUCKETS, registry=r)

    def observe_finished(self, seq) -> None:
        self.request_success_total.labels(seq.finish_reason or "unknown").inc()
        if seq.first_token_time is None or seq.finish_time is None:
            return
        self.time_to_first_token_seconds.observe(seq.first_token_time - seq.arrival_time)
        self.e2e_request_latency_seconds.observe(seq.finish_time - seq.arrival_time)
        if seq.num_output_tokens > 1:
            self.time_per_output_token_seconds.observe(
                (seq.finish_time - seq.first_token_time) / (seq.num_output_tokens - 1))

    def render(self) -> bytes:
        return generate_latest(self.registry)
