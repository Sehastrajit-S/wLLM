import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.embeddings import cosine_similarity, embed_text
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_native(MODEL_ID, dtype=torch.float32)
    return model, tokenizer


def test_embedding_is_l2_normalized(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    vec = embed_text(model, tokenizer, "The quick brown fox jumps over the lazy dog.")
    assert vec.shape == (model.cfg.hidden_size,)
    assert torch.isclose(vec.norm(), torch.tensor(1.0), atol=1e-4)


def test_identical_text_has_similarity_one(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    vec_a = embed_text(model, tokenizer, "What is the capital of France?")
    vec_b = embed_text(model, tokenizer, "What is the capital of France?")
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(1.0, abs=1e-4)


def test_similar_sentences_score_higher_than_dissimilar_ones(model_and_tokenizer):
    """The real sanity check: even a non-specialized embedder derived from a
    causal LM's own hidden states should separate related from unrelated
    text, not produce noise.
    """
    model, tokenizer = model_and_tokenizer
    anchor = embed_text(model, tokenizer, "What is the capital of France?")
    similar = embed_text(model, tokenizer, "Which city is the capital of France?")
    dissimilar = embed_text(model, tokenizer, "How do I bake a chocolate cake?")

    sim_similar = cosine_similarity(anchor, similar)
    sim_dissimilar = cosine_similarity(anchor, dissimilar)
    print(f"\nsimilar={sim_similar:.4f} dissimilar={sim_dissimilar:.4f}")
    assert sim_similar > sim_dissimilar
