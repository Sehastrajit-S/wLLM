"""Qwen2/Qwen2.5 architecture -- a thin subclass of the Llama-family base
implemented in llama.py, the same relationship real vLLM's own
`Qwen2Model(LlamaModel)` has to its `LlamaModel`. Qwen2's HF checkpoint
format was deliberately designed as a drop-in match for Llama's (RMSNorm,
RoPE, grouped-query attention, SwiGLU MLP, identical HF weight naming); the
one structural difference -- Qwen2 puts a bias on its q/k/v projections,
Llama doesn't -- is already read from config.json via
`ModelConfig.attention_bias` inside `LlamaAttention`, so no override is
needed here at all, only the class identity for `wllm.models.registry` to
dispatch on and for `isinstance` checks elsewhere (e.g. gguf_loader.py) to
target.

Every name below re-exports from llama.py purely for backward compatibility:
this module used to hold the only (Qwen2-named) copy of this architecture,
and a large fraction of the codebase (server, scripts, most of tests/)
imports `load_native`/`generate_*`/`RotaryEmbedding` from here specifically.
"""
from __future__ import annotations

from wllm.models.llama import (  # noqa: F401
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    generate_greedy_naive,
    generate_with_kv_cache,
    load_native,
    repeat_kv,
    rotate_half,
)


class Qwen2Model(LlamaModel):
    pass


class Qwen2ForCausalLM(LlamaForCausalLM):
    _model_cls = Qwen2Model
