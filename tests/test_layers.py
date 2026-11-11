"""Each layer matches the HuggingFace module on the same input (fp32, CPU)."""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from infer.kv.context import NoCacheContext
from infer.model.layers import apply_rope, rope_cache


def _causal_mask(n: int) -> torch.Tensor:
    m = torch.full((n, n), torch.finfo(torch.float32).min).triu(1)
    return m[None, None]


def test_rmsnorm_matches_hf(model_pair):
    ours, hf = model_pair
    torch.manual_seed(0)
    x = torch.randn(7, ours.cfg.hidden_size) * 3
    for i in (0, len(ours.layers) - 1):
        torch.testing.assert_close(ours.layers[i].input_layernorm(x), hf.model.layers[i].input_layernorm(x),
                                   rtol=0, atol=0)
    torch.testing.assert_close(ours.norm(x), hf.model.norm(x), rtol=0, atol=0)


def test_rmsnorm_fp16_casts_like_hf(model_pair):
    """The weight is applied after casting back to fp16, exactly as HF does."""
    ours, hf = model_pair
    torch.manual_seed(0)
    x = (torch.randn(5, ours.cfg.hidden_size) * 4).half()
    a = ours.norm.half()(x)
    b = hf.model.norm.half()(x)
    ours.norm.float(), hf.model.norm.float()
    assert a.dtype == torch.float16
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rope_matches_hf(model_pair, dtype):
    ours, hf = model_pair
    cfg = ours.cfg
    torch.manual_seed(0)
    positions = torch.tensor([0, 1, 5, 17, 300, 999])               # absolute, not contiguous
    q = torch.randn(len(positions), cfg.num_attention_heads, cfg.head_dim, dtype=dtype)
    k = torch.randn(len(positions), cfg.num_key_value_heads, cfg.head_dim, dtype=dtype)
    cos, sin = rope_cache(1000, cfg.head_dim, cfg.rope_theta)
    q1, k1 = apply_rope(q, k, cos, sin, positions)
    hcos, hsin = hf.model.rotary_emb(q.transpose(0, 1)[None], positions[None])   # [1, T, D] in dtype
    q2, k2 = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], hcos, hsin)
    torch.testing.assert_close(q1, q2[0].transpose(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(k1, k2[0].transpose(0, 1), rtol=0, atol=0)


def test_attention_matches_hf(model_pair):
    ours, hf = model_pair
    torch.manual_seed(0)
    n = 11
    x = torch.randn(n, ours.cfg.hidden_size)
    pos = torch.arange(n)
    for i in (0, len(ours.layers) - 1):
        mine = ours.layers[i].self_attn(x, pos, ours.rope_cos, ours.rope_sin,
                                        NoCacheContext([n], ours.cfg.head_dim ** -0.5))
        pe = hf.model.rotary_emb(x[None], pos[None])
        ref, _ = hf.model.layers[i].self_attn(x[None], pe, _causal_mask(n))
        torch.testing.assert_close(mine, ref[0], rtol=1e-4, atol=2e-5)


def test_mlp_matches_hf(model_pair):
    ours, hf = model_pair
    torch.manual_seed(0)
    x = torch.randn(9, ours.cfg.hidden_size)
    torch.testing.assert_close(ours.layers[0].mlp(x), hf.model.layers[0].mlp(x), rtol=1e-4, atol=2e-5)


def test_decoder_layer_matches_hf(model_pair):
    ours, hf = model_pair
    torch.manual_seed(0)
    n = 6
    x = torch.randn(n, ours.cfg.hidden_size)
    pos = torch.arange(n)
    mine = ours.layers[1](x, pos, ours.rope_cos, ours.rope_sin, NoCacheContext([n], ours.cfg.head_dim ** -0.5))
    pe = hf.model.rotary_emb(x[None], pos[None])
    ref = hf.model.layers[1](x[None], attention_mask=_causal_mask(n), position_embeddings=pe)
    ref = ref[0] if isinstance(ref, tuple) else ref
    torch.testing.assert_close(mine, ref[0], rtol=1e-4, atol=5e-5)


def test_output_layer_is_tied(model_pair):
    ours, _ = model_pair
    assert ours.cfg.tie_word_embeddings
    assert ours.output_weight.data_ptr() == ours.embed_tokens.weight.data_ptr()
