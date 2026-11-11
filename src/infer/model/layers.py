"""Qwen2 building blocks: RMSNorm, rotary embeddings, grouped-query attention, SwiGLU MLP.

Every cast mirrors HuggingFace's Qwen2 implementation so fp16 runs drift from it only where
kernels sum in a different order, never because of a different cast.

Tensors inside a step are flat: [tokens, hidden] or [tokens, heads, head_dim]. Tokens from many
sequences sit side by side, and an attention context (see infer.kv.context) knows which tokens
belong to which sequence and where their keys and values live in the KV cache.
"""

from __future__ import annotations

import os
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from infer.config import ModelConfig

# "sdpa" uses torch's fused kernels; "eager" is the explicit matmul-softmax-matmul HF uses with
# attn_implementation="eager". Both give the same greedy tokens in fp32; eager is for debugging.
ATTN_IMPL = os.environ.get("INFER_ATTN", "sdpa")


class AttentionContext(Protocol):
    """Writes this step's keys/values to the cache and runs attention for every sequence."""

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q: [T, H, D], k/v: [T, KVH, D] (RoPE already applied). Returns [T, H, D]."""
        ...


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()                                                  # compute in fp32
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)                             # cast back BEFORE the weight, as HF does


def rope_cache(max_pos: int, dim: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 cos/sin tables of shape [max_pos, dim], computed exactly as HF computes them per step."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q, k, cos, sin, positions):
    """q: [T, H, D], k: [T, KVH, D]; positions: [T] absolute position of each token in its own sequence."""
    c = cos[positions].to(q.dtype).unsqueeze(1)   # cast to the model dtype first, as HF does;
    s = sin[positions].to(q.dtype).unsqueeze(1)   # fp32 here would silently upcast q and k
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


# ---------------------------------------------------------------------------------------------
# Attention math. q/k/v arrive as [*, tokens, heads, D]; GQA is handled here so callers never
# repeat K/V themselves.
# ---------------------------------------------------------------------------------------------

def _attention(q, k, v, mask=None, is_causal=False, scale=None):
    """q/k/v: [B, heads, L, D] with equal head counts. mask: bool, True = may attend."""
    if ATTN_IMPL == "eager":
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if is_causal:
            lq, lk = q.shape[-2], k.shape[-2]
            mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(lk - lq)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs, v)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale)


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=dim)


def causal_attention(q, k, v, scale: float) -> torch.Tensor:
    """One sequence's prefill over its own fresh K/V. q: [L, H, D], k/v: [L, KVH, D] -> [L, H, D]."""
    groups = q.shape[1] // k.shape[1]
    qh = q.transpose(0, 1).unsqueeze(0)                                  # [1, H, L, D]
    kh = repeat_kv(k, groups, 1).transpose(0, 1).unsqueeze(0)
    vh = repeat_kv(v, groups, 1).transpose(0, 1).unsqueeze(0)
    out = _attention(qh, kh, vh, is_causal=True, scale=scale)
    return out[0].transpose(0, 1)


def masked_attention(q, k, v, mask, scale: float) -> torch.Tensor:
    """General case. q: [B, Lq, H, D], k/v: [B, Lk, KVH, D], mask: [B, Lq, Lk] bool -> [B, Lq, H, D]."""
    groups = q.shape[2] // k.shape[2]
    qh = q.transpose(1, 2)                                               # [B, H, Lq, D]
    kh = repeat_kv(k, groups, 2).transpose(1, 2)
    vh = repeat_kv(v, groups, 2).transpose(1, 2)
    out = _attention(qh, kh, vh, mask=mask.unsqueeze(1), scale=scale)
    return out.transpose(1, 2)


def decode_attention(q, k, v, context_lens, scale: float) -> torch.Tensor:
    """One new token per sequence. q: [B, H, D], k/v: [B, L, KVH, D] gathered from the cache,
    context_lens: [B]. Positions at or beyond a sequence's own length are masked.

    GQA without copying K/V: the 7 query heads that share a KV head become 7 "query rows" of a
    single-head attention over that KV head, so K and V are read once instead of 7 times.
    """
    b, h, d = q.shape
    kvh = k.shape[2]
    qg = q.view(b, kvh, h // kvh, d)                                    # [B, KVH, G, D]
    kh = k.transpose(1, 2)                                               # [B, KVH, L, D]
    vh = v.transpose(1, 2)
    mask = torch.arange(k.shape[1], device=q.device)[None, :] < context_lens[:, None]
    out = _attention(qg, kh, vh, mask=mask[:, None, None, :], scale=scale)   # [B, KVH, G, D]
    return out.reshape(b, h, d)


class Qwen2Attention(nn.Module):
    """Q, K and V are fused into one [hidden -> (H + 2*KVH) * D] matmul (fewer kernel launches)."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=True)
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=False)

    def forward(self, x, positions, cos, sin, ctx: AttentionContext) -> torch.Tensor:
        t = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)
        q, k = apply_rope(q, k, cos, sin, positions)
        out = ctx.attend(self.layer_idx, q, k, v)                       # [T, H, D]
        return self.o_proj(out.reshape(t, self.q_size))


class Qwen2MLP(nn.Module):
    """SwiGLU with gate and up fused into one [hidden -> 2 * intermediate] matmul."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.intermediate_size = cfg.intermediate_size
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.intermediate_size, dim=-1)
        return self.down_proj(F.silu(gate) * up)
