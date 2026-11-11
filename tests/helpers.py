"""Config builders shared by the tests."""

from __future__ import annotations

from infer.config import EngineConfig

REAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def tiny_cfg(**kw) -> EngineConfig:
    base = dict(model="tiny", device="cpu", dtype="float32", kv_cache_mb=4, block_size=4, max_model_len=256,
                max_num_seqs=32, max_num_batched_tokens=512, request_timeout=None, watermark=0.0)
    base.update(kw)
    return EngineConfig(**base)


def real_cfg(**kw) -> EngineConfig:
    base = dict(model=REAL_MODEL, device="cpu", dtype="float32", kv_cache_mb=512, block_size=16,
                max_model_len=2048, max_num_seqs=64, max_num_batched_tokens=4096, request_timeout=None)
    base.update(kw)
    return EngineConfig(**base)
