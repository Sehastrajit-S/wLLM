"""Gemma (1) architecture -- the first family this codebase supports that
genuinely diverges from the Llama recipe rather than just adding a config
flag. Reuses LlamaAttention as-is (RoPE, grouped-query attention, the shared
SDPA/PagedAttention forward paths -- Gemma's attention math itself, including
the paged KV cache and CUDA graph decode, is identical to Llama's given the
right head_dim), and only replaces what's actually different:

- RMSNorm uses a (1 + weight) parameterization instead of a bare weight
  (GemmaRMSNorm below), computed in float32 before the final downcast --
  same precision-sensitive ordering as real HF Gemma, not just the final
  fp32-only test value.
- MLP is gated, like Llama's SwiGLU, but with tanh-approximate GELU instead
  of SiLU (GemmaMLP below).
- Embeddings are scaled by hidden_size**0.5 immediately after lookup (set via
  LlamaModel's embed_scale hook -- see llama.py).
- head_dim is NOT hidden_size // num_attention_heads for real Gemma
  checkpoints (e.g. gemma-7b: hidden_size=3072, num_attention_heads=16, but
  head_dim=256) -- it's read explicitly from config.json via
  ModelConfig.explicit_head_dim.

Deliberately NOT covered here: Gemma 2 and 3. Both add mechanics this file's
approach can't just configure away -- attention logit softcapping (which
`torch.nn.functional.scaled_dot_product_attention` has no parameter for, so
it would need a hand-rolled attention path, and the custom PagedAttention
CUDA decode kernel has no notion of softcapping at all), alternating
sliding-window/full-attention layers, and (Gemma 2) sandwich normalization
around both sub-blocks. Supporting those for real -- not just prefill, but
through the same paged decode + CUDA graph path every other architecture
here gets -- is real kernel work, not a new model file.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from wllm.models.config import ModelConfig
from wllm.models.llama import LlamaAttention, LlamaForCausalLM, LlamaModel, RMSNorm


class GemmaRMSNorm(RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # (1 + weight), not weight -- and multiplied while still in float32,
        # matching real Gemma's modeling code exactly (the downcast happens
        # after this multiply, not before).
        x = x * (1.0 + self.weight.float())
        return x.to(dtype)


class GemmaMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):
    """Same pre-norm residual structure as LlamaDecoderLayer (two norms, not
    Gemma 2's four), and every one of LlamaDecoderLayer's forward variants
    (naive, prefill, chunked-prefill-with-prefix-reuse, decode, graphable
    decode) carries over unchanged -- attention itself is LlamaAttention,
    untouched, so Gemma gets the exact same paged KV cache and CUDA graph
    decode support Llama/Qwen2 get. Only the norm and MLP classes actually
    differ, so only those two lines differ from LlamaDecoderLayer in each
    method below.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = LlamaAttention(cfg)
        self.mlp = GemmaMLP(cfg)
        self.input_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

    def forward_prefill(self, x, cos, sin, k_cache, v_cache, positions):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn.forward_prefill(x, cos, sin, k_cache, v_cache, positions)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

    def forward_prefill_with_prefix(
        self, x, cos, sin, k_cache, v_cache, prefix_kv, positions, layer_idx=None, lora_ids=None, lora_registry=None
    ):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn.forward_prefill_with_prefix(
            x, cos, sin, k_cache, v_cache, prefix_kv, positions, layer_idx=layer_idx, lora_ids=lora_ids, lora_registry=lora_registry
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

    def forward_decode(self, x, cos, sin, k_cache, v_cache, positions, block_table, context_len, layer_idx=None, lora_ids=None, lora_registry=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn.forward_decode(
            x, cos, sin, k_cache, v_cache, positions, block_table, context_len,
            layer_idx=layer_idx, lora_ids=lora_ids, lora_registry=lora_registry,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

    def forward_decode_graphable(self, x, cos, sin, k_cache, v_cache, k_flat, v_flat, write_idx, block_table, context_len):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn.forward_decode_graphable(
            x, cos, sin, k_cache, v_cache, k_flat, v_flat, write_idx, block_table, context_len
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class GemmaModel(LlamaModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.layers = nn.ModuleList([GemmaDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.embed_scale = cfg.hidden_size**0.5
        # Re-registers the buffer LlamaModel.__init__ already created (still
        # holding the base class's default 1.0) with the real Gemma value --
        # see LlamaModel.__init__ for why this needs to be a buffer at all.
        self.register_buffer("_embed_scale_tensor", torch.tensor(self.embed_scale), persistent=False)


class GemmaForCausalLM(LlamaForCausalLM):
    """Full parity with Llama/Qwen2: every LlamaForCausalLM serving method
    (prefill_with_cache, continue_prefill + prefix reuse, decode_step_batch,
    decode_step_graphable) is inherited unchanged and works correctly here,
    since GemmaDecoderLayer implements the same forward_prefill/forward_decode/
    forward_decode_graphable methods LlamaDecoderLayer does.
    """

    _model_cls = GemmaModel
