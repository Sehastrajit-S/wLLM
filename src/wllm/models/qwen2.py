"""Native Qwen2/Llama-family forward pass, written directly on top of torch --
independent of `transformers`' modeling code. Parameter names mirror the HF
checkpoint layout 1:1 (model.embed_tokens, model.layers.N.self_attn.*, etc.)
so safetensors weights load straight in with `strict=True`.

Attention uses `torch.nn.functional.scaled_dot_product_attention` rather than
flash-attn (no official Windows wheels) -- this is PyTorch's own built-in
flash-attention-class kernel, compiled into the official Windows CUDA wheel.
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from wllm.engine.lora import apply_lora_delta
from wllm.models.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (batch, seq)
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq.to(position_ids.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos/sin: (batch, seq, head_dim) -> (batch, 1, seq, head_dim) for broadcast over heads
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kv_heads, seq, head_dim = x.shape
    x = x[:, :, None, :, :].expand(b, kv_heads, n_rep, seq, head_dim)
    return x.reshape(b, kv_heads * n_rep, seq, head_dim)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        # The one structural difference between Qwen2 and Llama this native
        # implementation otherwise shares in full (RMSNorm, RoPE, GQA,
        # SwiGLU MLP, HF weight naming convention -- Qwen2's HF checkpoint
        # format was deliberately designed as a drop-in match for Llama's):
        # Qwen2 adds a bias to q/k/v; Llama doesn't. o_proj has no bias in
        # either.
        qkv_bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape

        q = self.q_proj(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).reshape(b, seq, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)

    def forward_prefill(self, x, cos, sin, k_cache, v_cache, positions):
        """Computes attention normally (contiguous, causal) and writes the
        resulting K/V into the paged cache as a side effect, for later decode
        steps to read from.
        """
        b, seq, _ = x.shape

        q = self.q_proj(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_write = k[0].transpose(0, 1).float()  # (seq, num_kv_heads, head_dim)
        v_write = v[0].transpose(0, 1).float()
        for i, (block_id, slot) in enumerate(positions):
            k_cache[block_id, slot] = k_write[i]
            v_cache[block_id, slot] = v_write[i]

        k_full = repeat_kv(k, self.n_rep)
        v_full = repeat_kv(v, self.n_rep)
        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)

        attn_out = attn_out.transpose(1, 2).reshape(b, seq, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)

    def forward_prefill_with_prefix(
        self, x, cos, sin, k_cache, v_cache, prefix_kv, positions, layer_idx=None, lora_ids=None, lora_registry=None
    ):
        """Like forward_prefill, but x holds only the NEW suffix tokens --
        prefix_kv is (k, v) each (prefix_len, num_kv_heads, head_dim),
        already-cached K/V reused from an earlier sequence sharing this exact
        prefix (or None if there's no reused prefix at all). Each suffix
        query attends to the *entire* prefix (unconditionally -- it's fixed,
        already-computed past) plus itself causally, via an explicit mask
        rather than `is_causal=True` (which would wrongly forbid attending
        to the prefix at all).

        `lora_ids`/`lora_registry`, if given, apply a per-row LoRA delta
        (see engine/lora.py) to q/k/v/o projections only -- MLP LoRA targets
        aren't supported in this pass.
        """
        b, suffix_len, _ = x.shape
        assert b == 1

        q = self.q_proj(x)
        q = apply_lora_delta(x, q, lora_ids, lora_registry, layer_idx, "q_proj")
        k = self.k_proj(x)
        k = apply_lora_delta(x, k, lora_ids, lora_registry, layer_idx, "k_proj")
        v = self.v_proj(x)
        v = apply_lora_delta(x, v, lora_ids, lora_registry, layer_idx, "v_proj")

        q = q.view(b, suffix_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, suffix_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, suffix_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_write = k[0].transpose(0, 1).float()
        v_write = v[0].transpose(0, 1).float()
        for i, (block_id, slot) in enumerate(positions):
            k_cache[block_id, slot] = k_write[i]
            v_cache[block_id, slot] = v_write[i]

        if prefix_kv is not None:
            prefix_k, prefix_v = prefix_kv
            prefix_len = prefix_k.shape[0]
            # (prefix_len, num_kv_heads, head_dim) -> (1, num_kv_heads, prefix_len, head_dim), matching k/v's pre-repeat_kv layout
            prefix_k = prefix_k.unsqueeze(0).transpose(1, 2).to(k.dtype)
            prefix_v = prefix_v.unsqueeze(0).transpose(1, 2).to(v.dtype)
            k_all = torch.cat([prefix_k, k], dim=2)
            v_all = torch.cat([prefix_v, v], dim=2)

            mask = torch.zeros(suffix_len, prefix_len + suffix_len, dtype=torch.bool, device=x.device)
            mask[:, :prefix_len] = True
            mask[:, prefix_len:] = torch.tril(torch.ones(suffix_len, suffix_len, dtype=torch.bool, device=x.device))

            k_full = repeat_kv(k_all, self.n_rep)
            v_full = repeat_kv(v_all, self.n_rep)
            attn_out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)
        else:
            k_full = repeat_kv(k, self.n_rep)
            v_full = repeat_kv(v, self.n_rep)
            attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)

        attn_out = attn_out.transpose(1, 2).reshape(b, suffix_len, self.num_heads * self.head_dim)
        o = self.o_proj(attn_out)
        return apply_lora_delta(attn_out, o, lora_ids, lora_registry, layer_idx, "o_proj")

    def forward_decode(self, x, cos, sin, k_cache, v_cache, positions, block_table, context_len, layer_idx=None, lora_ids=None, lora_registry=None):
        """x holds one new token's embedding per sequence in the batch:
        (num_seqs, 1, hidden). `positions` is one (block_id, slot) per
        sequence (row-aligned). Writes each row's K/V into the cache then
        runs the custom PagedAttention CUDA kernel once for the whole batch
        (it already supports multiple sequences with independent context
        lengths -- validated in the P1.3 kernel tests).

        `lora_ids`/`lora_registry`: see forward_prefill_with_prefix. Not
        supported by forward_decode_graphable (CUDA graph capture bakes in a
        fixed computation at capture time; per-row adapter selection varying
        between replays doesn't fit that without its own dedicated static-
        buffer treatment, not implemented here) -- a graph-decoded sequence
        with a lora_id set falls back to this eager path instead.
        """
        from wllm.kernels.paged_attention import paged_attention_decode

        b = x.shape[0]
        q = self.q_proj(x)
        q = apply_lora_delta(x, q, lora_ids, lora_registry, layer_idx, "q_proj")
        k = self.k_proj(x)
        k = apply_lora_delta(x, k, lora_ids, lora_registry, layer_idx, "k_proj")
        v = self.v_proj(x)
        v = apply_lora_delta(x, v, lora_ids, lora_registry, layer_idx, "v_proj")

        q = q.view(b, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        for i, (block_id, slot) in enumerate(positions):
            k_cache[block_id, slot] = k[i, :, 0, :].float()
            v_cache[block_id, slot] = v[i, :, 0, :].float()

        q_flat = q[:, :, 0, :].reshape(b, self.num_heads, self.head_dim).float()
        scale = 1.0 / (self.head_dim**0.5)
        out = paged_attention_decode(q_flat, k_cache, v_cache, block_table, context_len, scale)

        out = out.to(x.dtype).reshape(b, 1, self.num_heads * self.head_dim)
        o = self.o_proj(out)
        return apply_lora_delta(out, o, lora_ids, lora_registry, layer_idx, "o_proj")

    def forward_decode_graphable(self, x, cos, sin, k_cache, v_cache, k_flat, v_flat, write_idx, block_table, context_len):
        """Same computation as forward_decode, but the cache write uses a
        tensor-valued index (index_copy_) instead of Python-int indexing.

        This distinction is what makes CUDA graph capture correct: Python-int
        indexing (`k_cache[block_id, slot] = ...`) resolves to a fixed memory
        address at *trace* time, so a captured graph would always overwrite
        the same slot on every replay. `index_copy_` reads the index
        *tensor's contents* at kernel *execution* time, so updating that
        tensor's values before each replay (which is the whole CUDA graph
        pattern: fixed buffers, changing contents) writes the correct new
        slot each time. k_flat/v_flat are (num_blocks*block_size, num_kv_heads,
        head_dim) views over the same storage as k_cache/v_cache -- writes
        through one are visible through the other.
        """
        from wllm.kernels.paged_attention import paged_attention_decode

        b = x.shape[0]
        q = self.q_proj(x).view(b, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_flat.index_copy_(0, write_idx, k[:, :, 0, :].float())
        v_flat.index_copy_(0, write_idx, v[:, :, 0, :].float())

        q_flat = q[:, :, 0, :].reshape(b, self.num_heads, self.head_dim).float()
        scale = 1.0 / (self.head_dim**0.5)
        out = paged_attention_decode(q_flat, k_cache, v_cache, block_table, context_len, scale)

        out = out.to(x.dtype).reshape(b, 1, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

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


class Qwen2Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, seq = input_ids.shape
        x = self.embed_tokens(input_ids)

        position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(b, -1)
        cos, sin = self.rotary_emb(position_ids)

        for layer in self.layers:
            x = layer(x, cos, sin)

        return self.norm(x)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen2Model(cfg)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids)
        return self._compute_logits(hidden)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final-layer hidden states (post-norm, pre-LM-head) -- what
        embedding/rerank want (a representation), as opposed to forward()'s
        next-token logits.
        """
        return self.model(input_ids)

    def _compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.cfg.tie_word_embeddings:
            return F.linear(hidden, self.model.embed_tokens.weight)
        return self.lm_head(hidden)

    def prefill_with_cache(self, cache, seq_id, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (1, prompt_len). Registers `seq_id` with the cache and
        writes every prefill position's K/V into it for later decode steps.
        """
        b, seq = input_ids.shape
        assert b == 1
        cache.create_sequence(seq_id)

        x = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        cos, sin = self.model.rotary_emb(position_ids)

        positions = cache.reserve(seq_id, seq)
        for i, layer in enumerate(self.model.layers):
            x = layer.forward_prefill(x, cos, sin, cache.k_caches[i], cache.v_caches[i], positions)
        cache.advance(seq_id, seq)

        x = self.model.norm(x)
        return self._compute_logits(x)

    def continue_prefill(self, cache, seq_id, chunk_token_ids: list, lora_id=None, lora_registry=None) -> torch.Tensor:
        """Processes exactly `chunk_token_ids` -- the next slice of a
        sequence's prompt -- attending to whatever's already cached for this
        seq_id (nothing, a reused prefix, one or more previous chunks, or
        any combination) as fixed non-causal context, causal only within
        this new chunk. `cache.create_sequence[_from_prefix]` must already
        have been called for seq_id. Call repeatedly with successive chunks
        to prefill a long prompt over multiple steps instead of all at once
        (chunked prefill), or once with the whole remaining prompt (as
        prefill_with_cache_and_prefix_reuse does) -- the math is identical
        either way, since "attend to whatever's cached so far" doesn't care
        how many calls built that state up.
        """
        already_cached = cache.context_len(seq_id)
        prefix_block_ids = cache.sequences[seq_id].block_table if already_cached > 0 else []

        chunk_len = len(chunk_token_ids)
        device = next(self.parameters()).device
        input_ids = torch.tensor([chunk_token_ids], device=device)

        x = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(already_cached, already_cached + chunk_len, device=device).unsqueeze(0)
        cos, sin = self.model.rotary_emb(position_ids)

        lora_ids = [lora_id]
        positions = cache.reserve(seq_id, chunk_len)
        for i, layer in enumerate(self.model.layers):
            prefix_kv = cache.gather_prefix_kv(i, prefix_block_ids, already_cached) if already_cached > 0 else None
            x = layer.forward_prefill_with_prefix(
                x, cos, sin, cache.k_caches[i], cache.v_caches[i], prefix_kv, positions,
                layer_idx=i, lora_ids=lora_ids, lora_registry=lora_registry,
            )
        cache.advance(seq_id, chunk_len)

        x = self.model.norm(x)
        return self._compute_logits(x)

    def prefill_with_cache_and_prefix_reuse(
        self, cache, seq_id, prompt_token_ids: list, lora_id=None, lora_registry=None
    ) -> torch.Tensor:
        """Like prefill_with_cache, but consults the cache's prefix-cache
        index first (a no-op unless the cache has enable_prefix_caching=True
        and something in it matches). Any whole blocks at the start of
        prompt_token_ids that exactly match an already-cached prefix are
        reused (refcounted, not recomputed or rewritten) -- only the
        remaining suffix actually runs through the model, in one shot (see
        the scheduler's chunked-prefill path for spreading that suffix
        across multiple steps instead). Registers any newly-completed blocks
        for future reuse once done.
        """
        matched_blocks, num_matched = cache.match_prefix(prompt_token_ids)

        if matched_blocks:
            cache.create_sequence_from_prefix(seq_id, matched_blocks)
        else:
            cache.create_sequence(seq_id)

        suffix_ids = prompt_token_ids[num_matched:]
        logits = self.continue_prefill(cache, seq_id, suffix_ids, lora_id=lora_id, lora_registry=lora_registry)

        cache.register_prefix_blocks(seq_id, prompt_token_ids)
        return logits

    def decode_step_with_cache(self, cache, seq_id, token_id: torch.Tensor) -> torch.Tensor:
        """Single-sequence convenience wrapper around decode_step_batch."""
        return self.decode_step_batch(cache, [seq_id], token_id)

    def decode_step_batch(self, cache, seq_ids: list, token_ids: torch.Tensor, lora_ids=None, lora_registry=None) -> torch.Tensor:
        """token_ids: (num_seqs, 1), each row the previous token for that
        sequence. Every sequence in the batch takes exactly one decode step,
        batched through a single PagedAttention kernel launch per layer.
        """
        device = token_ids.device
        x = self.model.embed_tokens(token_ids)  # (num_seqs, 1, hidden)

        old_lens = [cache.context_len(s) for s in seq_ids]
        position_ids = torch.tensor([[length] for length in old_lens], device=device)
        cos, sin = self.model.rotary_emb(position_ids)

        positions = [cache.reserve(s, 1)[0] for s in seq_ids]
        block_table = cache.block_tables_tensor(seq_ids, device=device)
        context_len_tensor = torch.tensor([length + 1 for length in old_lens], dtype=torch.int32, device=device)

        for i, layer in enumerate(self.model.layers):
            x = layer.forward_decode(
                x, cos, sin, cache.k_caches[i], cache.v_caches[i], positions, block_table, context_len_tensor,
                layer_idx=i, lora_ids=lora_ids, lora_registry=lora_registry,
            )
        for s in seq_ids:
            cache.advance(s, 1)

        x = self.model.norm(x)
        return self._compute_logits(x)

    def decode_step_graphable(self, cache, token_ids, position_ids, block_table, context_len, write_idx):
        """The tensor-computation-only core of a decode step, built entirely
        from pre-existing static buffers -- no Python-side cache bookkeeping
        (reserve/advance, building lists into fresh tensors) happens in here,
        since none of that can be captured by a CUDA graph anyway. The
        caller (CUDAGraphDecoder) is responsible for populating these buffers
        with real values before every real call/replay.
        """
        x = self.model.embed_tokens(token_ids)  # (batch, 1, hidden)
        cos, sin = self.model.rotary_emb(position_ids)

        for i, layer in enumerate(self.model.layers):
            k_cache = cache.k_caches[i]
            v_cache = cache.v_caches[i]
            num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
            k_flat = k_cache.view(num_blocks * block_size, num_kv_heads, head_dim)
            v_flat = v_cache.view(num_blocks * block_size, num_kv_heads, head_dim)
            x = layer.forward_decode_graphable(x, cos, sin, k_cache, v_cache, k_flat, v_flat, write_idx, block_table, context_len)

        x = self.model.norm(x)
        return self._compute_logits(x)


@torch.inference_mode()
def generate_with_kv_cache(model, cache, seq_id, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    logits = model.prefill_with_cache(cache, seq_id, input_ids)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token]

    for _ in range(max_new_tokens - 1):
        logits = model.decode_step_with_cache(cache, seq_id, next_token)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token)

    return torch.cat([input_ids] + tokens, dim=1)


@torch.inference_mode()
def generate_greedy_naive(model: Qwen2ForCausalLM, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Recomputes the full forward pass every step (no KV cache) -- purely a
    correctness sanity check for the native model runner. Real generation
    (with the PagedAttention KV cache) lands in P1.3.
    """
    tokens = input_ids
    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


def _find_checkpoint_dir(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"])


def load_native(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Qwen2ForCausalLM:
    ckpt_dir = _find_checkpoint_dir(model_id)

    cfg = ModelConfig.from_json(os.path.join(ckpt_dir, "config.json"))
    model = Qwen2ForCausalLM(cfg)

    state_dict = {}
    for shard_path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        state_dict.update(load_file(shard_path))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [m for m in missing if not (cfg.tie_word_embeddings and m == "lm_head.weight")]
    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model
