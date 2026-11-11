"""The full Qwen2 decoder: embedding -> N x (attention, MLP) -> final norm -> tied output layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from infer.config import ModelConfig
from infer.model.layers import AttentionContext, Qwen2Attention, Qwen2MLP, RMSNorm, rope_cache


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen2Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen2MLP(cfg)

    def forward(self, x, positions, cos, sin, ctx):
        x = x + self.self_attn(self.input_layernorm(x), positions, cos, sin, ctx)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_positions: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen2DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        n_pos = min(max_positions or cfg.max_position_embeddings, cfg.max_position_embeddings)
        cos, sin = rope_cache(n_pos, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)        # stays fp32; cast per use
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def output_weight(self) -> torch.Tensor:
        return self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ctx: AttentionContext) -> torch.Tensor:
        """input_ids, positions: [T] (tokens of every sequence in this step, side by side).
        Returns final-normed hidden states [T, hidden]; call compute_logits on the rows you sample."""
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, self.rope_cos, self.rope_sin, ctx)
        return self.norm(x)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.output_weight)


# ---------------------------------------------------------------------------------------------
# Part 2 reference path: one sequence, contiguous cache, prefill then decode one token at a time.
# The engine does not use this; tests use it to check the forward pass in isolation.
# ---------------------------------------------------------------------------------------------

class SingleSequenceCache:
    """Per layer, k_cache and v_cache of shape [max_len, KVH, D] for a single sequence."""

    def __init__(self, cfg: ModelConfig, max_len: int, dtype, device):
        shape = (cfg.num_hidden_layers, max_len, cfg.num_key_value_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0          # tokens stored so far
        self.step_len = 0        # tokens being added by the current forward
        self.scale = cfg.head_dim ** -0.5

    def attend(self, layer_idx, q, k, v):
        from infer.model.layers import causal_attention, decode_attention

        start, n = self.length, q.shape[0]
        self.k[layer_idx, start:start + n] = k
        self.v[layer_idx, start:start + n] = v
        if start == 0:
            return causal_attention(q, k, v, self.scale)
        assert n == 1, "reference cache decodes one token at a time"
        ctx_len = torch.tensor([start + 1], device=q.device)
        return decode_attention(q, self.k[layer_idx, None, :start + 1], self.v[layer_idx, None, :start + 1],
                                ctx_len, self.scale)


@torch.inference_mode()
def greedy_generate_reference(model: Qwen2ForCausalLM, prompt_ids: list[int], max_tokens: int,
                              stop_ids: set[int] | None = None) -> list[int]:
    """prefill -> sample -> decode loop for one sequence with argmax sampling."""
    device = model.embed_tokens.weight.device
    dtype = model.embed_tokens.weight.dtype
    cache = SingleSequenceCache(model.cfg, len(prompt_ids) + max_tokens, dtype, device)
    ids = torch.tensor(prompt_ids, device=device)
    hidden = model(ids, torch.arange(len(prompt_ids), device=device), cache)
    cache.length = len(prompt_ids)
    logits = model.compute_logits(hidden[-1:])                          # last position only
    out: list[int] = []
    for _ in range(max_tokens):
        tok = int(logits.argmax(-1))
        out.append(tok)
        if stop_ids and tok in stop_ids:
            break
        pos = len(prompt_ids) + len(out) - 1
        hidden = model(torch.tensor([tok], device=device), torch.tensor([pos], device=device), cache)
        cache.length += 1
        logits = model.compute_logits(hidden)
    return out
