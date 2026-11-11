"""Incremental detokenization: multi-byte characters never break, stop strings trim correctly."""

from __future__ import annotations

import pytest

from helpers import REAL_MODEL
from infer.detokenizer import IncrementalDetokenizer

TEXTS = [
    "Hello, world! This is plain ASCII text.",
    "用中文解释什么是机器学习，并举两个例子。",
    "मशीन लर्निंग क्या है? सरल शब्दों में समझाइए।",
    "Party time 🎉🎂🥳 with family 👨‍👩‍👧‍👦 and flags 🇮🇳🇯🇵!",
    "Mixed: naïve café — “quotes” and math ∑ x² ≤ ∞.",
]


@pytest.fixture(scope="module", params=["tiny", pytest.param("qwen", marks=pytest.mark.real_model)])
def tokenizer(request):
    if request.param == "tiny":
        return request.getfixturevalue("tiny_tokenizer")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(REAL_MODEL)
    except Exception as e:  # offline
        pytest.skip(f"no Qwen tokenizer: {e}")


@pytest.mark.parametrize("text", TEXTS)
def test_streamed_pieces_concatenate_to_full_decode(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    d = IncrementalDetokenizer(tokenizer)
    pieces = [d.add([t]) for t in ids]
    pieces.append(d.flush())
    assert "".join(pieces) == tokenizer.decode(ids) == text
    assert not any("�" in p for p in pieces), "a multi-byte character was split"


def test_one_token_at_a_time_would_break(tokenizer):
    """The naive approach the detokenizer exists to avoid: decoding tokens one by one."""
    ids = tokenizer.encode(TEXTS[3], add_special_tokens=False)
    naive = "".join(tokenizer.decode([t]) for t in ids)
    assert naive != TEXTS[3]


def test_special_tokens_skipped(tokenizer):
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = tokenizer.encode("done", add_special_tokens=False) + [eos]
    d = IncrementalDetokenizer(tokenizer)
    assert "".join(d.add([t]) for t in ids) + d.flush() == "done"


def test_stop_string_trimmed_and_never_half_streamed(tokenizer):
    text = "The answer is 42. END of story, more text follows here."
    ids = tokenizer.encode(text, add_special_tokens=False)
    d = IncrementalDetokenizer(tokenizer, stop=["END"])
    streamed = []
    for t in ids:
        streamed.append(d.add([t]))
        assert "E" not in "".join(streamed) or d.stopped        # 'E' of END held back until decided
        if d.stopped:
            break
    assert d.stopped
    assert "".join(streamed) == "The answer is 42. "
    assert tokenizer.decode(ids[:d.num_tokens_at_stop]).find("END") != -1
    assert tokenizer.decode(ids[:d.num_tokens_at_stop - 1]).find("END") == -1


def test_earliest_of_several_stop_strings(tokenizer):
    d = IncrementalDetokenizer(tokenizer, stop=["world", "lo"])
    out = d.add(tokenizer.encode("hello world", add_special_tokens=False)) + d.flush()
    assert out == "hel" and d.stopped


def test_no_stop_releases_everything_at_flush(tokenizer):
    d = IncrementalDetokenizer(tokenizer, stop=["zzzz"])
    ids = tokenizer.encode("abc def", add_special_tokens=False)
    out = d.add(ids)
    assert len(out) <= len("abc def") - 3                         # last 3 chars held back
    assert out + d.flush() == "abc def"
