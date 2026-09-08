"""Embeddings from a decoder-only causal LM: no specialized embedding head
exists on this model, so we use the standard technique for repurposing a
causal LM as an embedder -- last-token pooling (causal attention means the
final position has attended to the whole sequence, so its hidden state is a
reasonable whole-sequence representation) followed by L2 normalization
(standard for embeddings meant to be compared via cosine similarity/dot
product). This does NOT match the embedding quality of a model actually
trained for the task (e.g. e5-mistral, GritLM) -- it's the same
architecture's own hidden states repurposed, not a specialized embedder.

Deliberately unbatched (one forward pass per input text): batching would
need padding, and our native model's plain forward has no attention-mask
support anywhere in the codebase yet (every other path either has no padding
at all -- single-sequence prefill -- or uses the paged KV cache, which has
no notion of padding either). Adding that just for embeddings isn't worth
it yet; this is correctness-first, with batched+padded embeddings as a
clear, deferred, future optimization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def embed_text(model, tokenizer, text: str, device: str = "cuda") -> torch.Tensor:
    """Returns a single L2-normalized embedding vector, shape (hidden_size,)."""
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    hidden = model.hidden_states(input_ids)  # (1, seq, hidden)
    vec = hidden[0, -1, :]  # last-token pooling
    return F.normalize(vec, p=2, dim=-1)


@torch.inference_mode()
def embed_texts(model, tokenizer, texts: list[str], device: str = "cuda") -> list[torch.Tensor]:
    return [embed_text(model, tokenizer, t, device=device) for t in texts]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Both must already be L2-normalized (as embed_text's output is) --
    then cosine similarity is just the dot product.
    """
    return torch.dot(a, b).item()
