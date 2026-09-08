"""Minimal config parsing independent of `transformers.AutoConfig` -- just the
fields the native Qwen2/Llama-family forward pass needs, read directly from
the checkpoint's config.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str) -> ModelConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        rope_theta = raw.get("rope_theta")
        if rope_theta is None:
            rope_theta = raw.get("rope_parameters", {}).get("rope_theta", 10000.0)

        # Qwen2's HF config never sets this field at all (its q/k/v
        # projections always have a bias -- that's the one structural
        # difference from Llama, which does expose it, defaulting to
        # False). Absent the field, fall back on model_type rather than a
        # single hardcoded default, since a third architecture reusing this
        # loader is more likely to follow Llama's no-bias convention than
        # Qwen2's.
        attention_bias = raw.get("attention_bias")
        if attention_bias is None:
            attention_bias = raw.get("model_type") == "qwen2"

        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope_theta),
            max_position_embeddings=raw.get("max_position_embeddings", 4096),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=bool(attention_bias),
        )
