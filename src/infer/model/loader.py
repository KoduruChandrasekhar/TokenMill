"""Load Qwen2 weights straight from .safetensors files into our own modules.

Checkpoint names map onto our modules, except that q/k/v_proj are concatenated into qkv_proj and
gate/up_proj into gate_up_proj. The output layer is tied to the embedding when the config says so.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

from infer.config import ModelConfig
from infer.model.qwen2 import Qwen2ForCausalLM

_ALLOW = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"]


def resolve_model_path(model: str) -> Path:
    """A local directory as-is, otherwise download (or reuse the cached) snapshot from the HF hub."""
    if os.path.isdir(model):
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=_ALLOW))


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)")
_QKV = {"self_attn.q_proj": 0, "self_attn.k_proj": 1, "self_attn.v_proj": 2}
_GATE_UP = {"mlp.gate_proj": 0, "mlp.up_proj": 1}


@torch.no_grad()
def load_model(path: str | Path, dtype: torch.dtype, device: torch.device,
               max_positions: int | None = None) -> Qwen2ForCausalLM:
    path = Path(path)
    cfg = ModelConfig.from_pretrained_dir(path)
    with torch.device("meta"):
        model = Qwen2ForCausalLM(cfg, max_positions=max_positions)
    # Parameters take the requested dtype (meta tensors are created in the default fp32; copying
    # fp16 weights into them would silently run the whole model in fp32).
    model = model.to_empty(device=device).to(dtype).requires_grad_(False)
    # The rope tables were created on meta; rebuild them for real, and keep them fp32: they are
    # cast to the activation dtype at use, exactly as HF casts its fp32 cos/sin.
    from infer.model.layers import rope_cache

    model.rope_cos, model.rope_sin = rope_cache(model.rope_cos.shape[0], cfg.head_dim, cfg.rope_theta,
                                                device=device)

    params = dict(model.named_parameters())
    loaded: set[str] = set()
    q, kv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
    qkv_offsets = [0, q, q + kv, q + 2 * kv]
    inter = cfg.intermediate_size

    files = sorted(glob.glob(str(path / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name).to(dtype)
                target, rows = _map_name(name, qkv_offsets, inter)
                if target is None:
                    continue
                if target == "lm_head.weight" and model.lm_head is None:
                    continue                                   # tied: the embedding already holds it
                dst = params[target]
                if rows is None:
                    dst.copy_(tensor)
                else:
                    dst[rows[0]:rows[1]].copy_(tensor)
                loaded.add(f"{target}[{rows}]" if rows else target)

    expected = _expected_keys(model, qkv_offsets, inter)
    missing = expected - loaded
    if missing:
        raise RuntimeError(f"weights missing from checkpoint: {sorted(missing)[:8]} ...")
    return model.eval()


def _map_name(name: str, qkv_offsets: list[int], inter: int):
    """Checkpoint name -> (our parameter name, optional (row_start, row_end) inside a fused matrix)."""
    if name == "model.embed_tokens.weight":
        return "embed_tokens.weight", None
    if name == "model.norm.weight":
        return "norm.weight", None
    if name == "lm_head.weight":
        return "lm_head.weight", None
    m = _LAYER_RE.fullmatch(name)
    if not m:
        return None, None
    i, rest = m.group(1), m.group(2)
    module, _, kind = rest.rpartition(".")                # e.g. self_attn.q_proj , weight
    if module in _QKV:
        j = _QKV[module]
        return f"layers.{i}.self_attn.qkv_proj.{kind}", (qkv_offsets[j], qkv_offsets[j + 1])
    if module in _GATE_UP:
        j = _GATE_UP[module]
        return f"layers.{i}.mlp.gate_up_proj.{kind}", (j * inter, (j + 1) * inter)
    if "rotary_emb" in rest:
        return None, None
    return f"layers.{i}.{rest}", None


def _expected_keys(model: Qwen2ForCausalLM, qkv_offsets, inter) -> set[str]:
    keys = set()
    for name, _ in model.named_parameters():
        if ".qkv_proj." in name:
            keys |= {f"{name}[{(qkv_offsets[j], qkv_offsets[j + 1])}]" for j in range(3)}
        elif ".gate_up_proj." in name:
            keys |= {f"{name}[{(j * inter, (j + 1) * inter)}]" for j in range(2)}
        else:
            keys.add(name)
    return keys


def load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))
