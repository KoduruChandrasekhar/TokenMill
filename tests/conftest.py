"""Shared fixtures.

* tiny_*: a 2-layer Qwen2 with random weights and a small byte-level BPE tokenizer trained on the
  fly. No network, a few seconds to build: used by scheduler, engine, API and CI tests.
* real_*: Qwen2.5-0.5B-Instruct from the HF hub (cached after the first download), fp32 on CPU.
  Marked real_model; set INFER_SKIP_REAL=1 to skip them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
from helpers import REAL_MODEL  # noqa: E402

CHATML = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
    "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Continuous batching keeps the GPU busy; paged attention keeps memory tidy.",
    "def hello(name):\n    return f'hello {name}'\n",
    "Numbers like 12345 and 3.14159 appear in text, as do emoji 🎉 and 中文 and हिन्दी.",
    "Large language models predict the next token given all previous tokens.",
] * 20


def _build_tiny(path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(_CORPUS, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=["<|im_start|>"])
    fast.chat_template = CHATML
    fast.save_pretrained(path)
    im_end, eot = fast.convert_tokens_to_ids("<|im_end|>"), fast.convert_tokens_to_ids("<|endoftext|>")

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True,
                      initializer_range=0.2, eos_token_id=im_end, bos_token_id=eot, pad_token_id=eot)
    model = Qwen2ForCausalLM(cfg)
    with torch.no_grad():                    # HF zero-inits biases; give them values so bias bugs show
        for layer in model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.bias.normal_(0, 0.2)
            layer.input_layernorm.weight.uniform_(0.5, 1.5)
            layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    model.save_pretrained(path, safe_serialization=True)
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [im_end, eot]}))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tiny-qwen2")
    _build_tiny(path)
    return path


@pytest.fixture(scope="session")
def tiny_model(tiny_model_dir):
    from infer.model.loader import load_model

    return load_model(tiny_model_dir, torch.float32, torch.device("cpu"))


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_dir):
    from infer.model.loader import load_tokenizer

    return load_tokenizer(tiny_model_dir)


@pytest.fixture(scope="session")
def tiny_hf_model(tiny_model_dir):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32,
                                                attn_implementation="eager").eval()


# --- the real model -----------------------------------------------------------------------------

def _skip_real():
    if os.environ.get("INFER_SKIP_REAL"):
        pytest.skip("INFER_SKIP_REAL is set")


@pytest.fixture(scope="session")
def real_model_dir() -> Path:
    _skip_real()
    from infer.model.loader import resolve_model_path

    try:
        return resolve_model_path(REAL_MODEL)
    except Exception as e:  # offline and not cached
        pytest.skip(f"cannot get {REAL_MODEL}: {e}")


@pytest.fixture(scope="session")
def real_model(real_model_dir):
    from infer.model.loader import load_model

    return load_model(real_model_dir, torch.float32, torch.device("cpu"), max_positions=4096)


@pytest.fixture(scope="session")
def real_tokenizer(real_model_dir):
    from infer.model.loader import load_tokenizer

    return load_tokenizer(real_model_dir)


@pytest.fixture(scope="session")
def golden() -> dict:
    path = ROOT / "tests" / "golden.json"
    if not path.exists():
        pytest.skip("tests/golden.json missing: run python baselines/make_golden.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def real_hf_model(real_model_dir):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(real_model_dir, dtype=torch.float32,
                                                attn_implementation="eager").eval()


@pytest.fixture(params=["tiny", pytest.param("real", marks=pytest.mark.real_model)])
def model_pair(request):
    """(our model, HF model) built from the same checkpoint."""
    if request.param == "tiny":
        return request.getfixturevalue("tiny_model"), request.getfixturevalue("tiny_hf_model")
    return request.getfixturevalue("real_model"), request.getfixturevalue("real_hf_model")
