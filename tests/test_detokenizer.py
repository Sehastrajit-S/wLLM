import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from transformers import AutoTokenizer

from wllm.engine.detokenizer import IncrementalDetokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def test_incremental_deltas_reassemble_to_full_decode(tokenizer):
    text = "The capital of France is Paris. It has a population of over 2 million people!"
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    detok = IncrementalDetokenizer(tokenizer)
    reassembled = "".join(detok.add_token(t) for t in token_ids)

    expected = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert reassembled == expected


def test_incremental_deltas_reassemble_for_unicode_text(tokenizer):
    text = "Hello 你好 \U0001f600 café"
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    detok = IncrementalDetokenizer(tokenizer)
    reassembled = "".join(detok.add_token(t) for t in token_ids)

    expected = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert reassembled == expected


def test_special_tokens_are_skipped():
    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            words = {1: "hello", 2: "world", 99: "<eos>"}
            visible = [words[i] for i in ids if not (skip_special_tokens and i == 99)]
            return " ".join(visible)

    detok = IncrementalDetokenizer(FakeTokenizer())
    out = detok.add_token(1) + detok.add_token(2) + detok.add_token(99)
    assert "eos" not in out
    assert out == "hello world"
