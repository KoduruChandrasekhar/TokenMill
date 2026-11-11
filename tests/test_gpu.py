"""GPU-only checks (skipped without CUDA). Run on the T4 you benchmark on:

    pytest -m gpu tests/test_gpu.py

fp16 standard (testing strategy): kernels and batch sizes change the order in which numbers are
added, so a greedy token may flip wherever the top two logits are nearly tied. So: our fp16 logits
stay within the measured tolerance of HuggingFace fp16, and every token that differs from the fp32
golden output sits at a near-tie (top-2 gap below the tolerance).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from helpers import REAL_MODEL, real_cfg
from infer.engine.engine import LLMEngine
from infer.engine.sequence import SamplingParams
from infer.kv.context import NoCacheContext

pytestmark = [pytest.mark.gpu, pytest.mark.real_model,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

TOL_FILE = Path(__file__).parent / "fp16_tolerance.json"
# Two independent fp16 implementations each sit within the tolerance of the truth, so they sit
# within twice the tolerance of each other (triangle inequality).
TOL_FACTOR = float(os.environ.get("INFER_FP16_TOL_FACTOR", "2.0"))


@pytest.fixture(scope="module")
def fp16_model(real_model_dir):
    from infer.model.loader import load_model

    return load_model(real_model_dir, torch.float16, torch.device("cuda"), max_positions=4096)


@pytest.fixture(scope="module")
def tolerance() -> float:
    if not TOL_FILE.exists():
        pytest.skip("tests/fp16_tolerance.json missing: python baselines/make_golden.py --fp16-only --device cuda")
    doc = json.loads(TOL_FILE.read_text())
    if not str(doc.get("env", {}).get("device", "")).startswith("cuda"):
        pytest.skip("fp16 tolerance was measured on CPU; re-measure it on this GPU")
    return float(doc["tolerance"])


def _tf_logits(model, ids: list[int]) -> torch.Tensor:
    """Teacher-forced logits for every position of ids (fp32 copy)."""
    x = torch.tensor(ids, device="cuda")
    with torch.inference_mode():
        h = model(x, torch.arange(len(ids), device="cuda"), NoCacheContext.single(len(ids), model.cfg))
        return model.compute_logits(h).float()


def _check_near_ties(model, outs, cases, tol) -> list[str]:
    problems = []
    for i, (got, c) in enumerate(zip(outs, cases)):
        want = c["output_ids"]
        j = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), None)
        if j is None:
            continue
        logits = _tf_logits(model, c["prompt_ids"] + want[:j])[-1]
        top2 = logits.topk(2).values
        gap = float(top2[0] - top2[1])
        if gap > tol:
            problems.append(f"prompt {i} diverged at token {j} with top-2 gap {gap:.4f} > {tol:.4f}")
    return problems


def test_fp16_logits_within_tolerance(fp16_model, golden, tolerance):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(REAL_MODEL, dtype=torch.float16).cuda().eval()
    worst = 0.0
    for c in golden["cases"][:20]:
        ids = c["prompt_ids"] + c["output_ids"][:-1]
        n = len(c["output_ids"])
        mine = _tf_logits(fp16_model, ids)[-n:]
        with torch.inference_mode():
            ref = hf(torch.tensor([ids], device="cuda")).logits[0, -n:].float()
        worst = max(worst, float((mine - ref).abs().max()))
    assert worst <= TOL_FACTOR * tolerance, f"max |logit diff| {worst:.4f} vs tolerance {tolerance:.4f}"


def test_golden_fp16_diverges_only_at_near_ties(fp16_model, golden, tolerance):
    from infer.model.loader import load_tokenizer

    eng = LLMEngine(real_cfg(device="cuda", dtype="float16", kv_cache_mb=1024),
                    model=fp16_model, tokenizer=load_tokenizer(REAL_MODEL))
    cases = golden["cases"]
    outs = eng.generate([c["prompt_ids"] for c in cases], SamplingParams(max_tokens=golden["max_new_tokens"]))
    problems = _check_near_ties(fp16_model, outs, cases, tolerance)
    assert not problems, "; ".join(problems[:5])


def test_batch_invariance_fp16(fp16_model, golden, tolerance):
    """A prompt alone and inside a batch of 32: same tokens, except at near-ties."""
    eng = LLMEngine(real_cfg(device="cuda", dtype="float16", kv_cache_mb=1024), model=fp16_model)
    cases = golden["cases"][:32]
    p = SamplingParams(max_tokens=golden["max_new_tokens"])
    batch = eng.generate([c["prompt_ids"] for c in cases], p)
    for i in (0, 7, 31):
        alone = eng.generate([cases[i]["prompt_ids"]], p)[0]
        if alone != batch[i]:
            ref = {**cases[i], "output_ids": alone}
            assert not _check_near_ties(fp16_model, [batch[i]], [ref], tolerance)


def test_cuda_graphs_match_eager_fp32(real_model_dir, golden):
    """In fp32 the graph replay must give exactly the eager engine's greedy tokens."""
    from infer.model.loader import load_model

    model = load_model(real_model_dir, torch.float32, torch.device("cuda"), max_positions=4096)
    cases = golden["cases"][:24]
    p = SamplingParams(max_tokens=golden["max_new_tokens"])
    eager = LLMEngine(real_cfg(device="cuda", dtype="float32", kv_cache_mb=1024, max_num_seqs=32), model=model)
    graphs = LLMEngine(real_cfg(device="cuda", dtype="float32", kv_cache_mb=1024, max_num_seqs=32,
                                cuda_graphs=True), model=model)
    assert graphs.runner.graphs is not None and graphs.runner.graphs.graphs
    a = eager.generate([c["prompt_ids"] for c in cases], p)
    b = graphs.generate([c["prompt_ids"] for c in cases], p)
    assert a == b
    for got, c in zip(a, cases):                          # exact, except at a recorded fp32 near-tie
        if not c.get("fp32_near_ties"):
            assert got == c["output_ids"]


def test_cuda_graphs_fp16_logits_close(fp16_model, tolerance):
    from infer.engine.cuda_graphs import CudaGraphRunner

    eng = LLMEngine(real_cfg(device="cuda", dtype="float16", kv_cache_mb=512, max_num_seqs=16, cuda_graphs=True),
                    model=fp16_model)
    runner: CudaGraphRunner = eng.runner.graphs
    prompts = [[1000 + i * 7 + j for j in range(30 + 5 * i)] for i in range(5)]
    for i, pr in enumerate(prompts):
        eng.add_request(f"g{i}", pr, SamplingParams(max_tokens=50, ignore_eos=True))
    eng.step()                                                   # prefill all
    seqs = list(eng.scheduler.running)
    for s in seqs:
        eng.cache.append_slot(s)
    with torch.inference_mode():
        graph_logits = runner.run(seqs).float().clone()
        ids, pos, ctx, last = eng.runner.prepare([], seqs)
        eager_logits = eng.model.compute_logits(eng.model(ids, pos, ctx)[last]).float()
    assert float((graph_logits - eager_logits).abs().max()) <= TOL_FACTOR * tolerance
