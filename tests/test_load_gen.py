"""Load-generator metric definitions (no server needed)."""

from dataclasses import asdict

import pytest

from bench.load_gen import RequestRecord, summarize, summarize_raw


def _done(send: float, first: float, end: float, tokens: int = 11) -> RequestRecord:
    step = (end - first) / (tokens - 1)
    return RequestRecord(prompt_len=8, output_len=tokens, send_time=send,
                         chunk_times=[first + i * step for i in range(tokens)], end_time=end,
                         output_tokens=tokens, ok=True, status=200)


def test_metrics_of_a_plain_run():
    recs = [_done(100.0, 100.1, 101.1), _done(100.5, 100.6, 102.6)]
    s = summarize({"records": recs, "duration": 5.0, "hit_time_cap": False, "timeline": []})
    assert s["duration_s"] == pytest.approx(2.6)              # first send -> last token
    assert s["output_throughput_tok_s"] == pytest.approx(22 / 2.6)
    assert s["ttft_s"]["p50"] == pytest.approx(0.1)
    assert s["tpot_s"]["p50"] == pytest.approx(0.15)          # (1.0/10 + 2.0/10) / 2
    assert not s["saturated"] and not s["hit_time_cap"]


def test_time_cap_with_queued_requests():
    """Concurrency 1 hitting the cap: queued requests were never sent (send_time 0). They must not
    stretch the duration to the clock's origin, and hitting the cap is not saturation."""
    recs = [_done(1000.0, 1000.03, 1003.5), _done(1003.5, 1003.53, 1007.0)]
    recs += [RequestRecord(prompt_len=8, output_len=11, error="cancelled: run hit the time cap")
             for _ in range(3)]
    s = summarize({"records": recs, "duration": 7.0, "hit_time_cap": True, "timeline": []})
    assert s["duration_s"] == pytest.approx(7.0)
    assert s["output_throughput_tok_s"] == pytest.approx(22 / 7.0)
    assert s["completed"] == 2 and s["failed"] == 3
    assert s["hit_time_cap"] and not s["saturated"]


def test_queueing_is_saturation():
    recs = [_done(0.5, 40.0, 41.0)]
    s = summarize({"records": recs, "duration": 41.0, "hit_time_cap": False, "timeline": []})
    assert s["saturated"]


def test_summarize_raw_recomputes_from_stored_requests():
    recs = [_done(1000.0, 1000.03, 1003.5)]
    recs.append(RequestRecord(prompt_len=8, output_len=11, error="cancelled: run hit the time cap"))
    stale = {"duration_s": 900.0, "output_throughput_tok_s": 0.01, "saturated": True}
    doc = {"summary": stale, "timeline": [{"t": 0.0, "running": 1.0}],
           "requests": [{**asdict(r), "ttft": r.ttft, "tpot": r.tpot} for r in recs]}
    s = summarize_raw(doc)
    assert s["output_throughput_tok_s"] == pytest.approx(11 / 3.5)
    assert s["hit_time_cap"] and not s["saturated"]
    assert s["max_concurrent_seqs"] == 1.0
