"""Loads a Qwen2-family GGUF checkpoint into our native model: parses
llama.cpp's metadata/tensor naming conventions into our ModelConfig and
state-dict layout, dequantizes small F32/F16 tensors (embeddings, norms,
biases) eagerly, and swaps the big weight matrices' Linear modules for
QuantizedLinear where the GGUF tensor is actually quantized (Q4_0/Q8_0) --
keeping them compact in GPU memory rather than expanding everything to
fp16/bf16 at load time.
"""
from __future__ import annotations

import re

import gguf
import numpy as np
import torch

from wllm.models.config import ModelConfig
from wllm.models.qwen2 import Qwen2ForCausalLM, RotaryEmbedding
from wllm.quant.dequant import GGML_TYPE_F16, GGML_TYPE_F32, dequantize
from wllm.quant.quant_linear import QuantizedLinear

_BLOCK_TENSOR_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

# Linear submodules (dotted path relative to a decoder layer) that GGUF
# stores as one 2D matrix each -- these are the ones eligible for the
# QuantizedLinear swap. Everything else (norms, biases, embeddings) is
# always F32/F16 in GGUF and just gets dequantized straight into the model.
_LINEAR_SUFFIXES = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}


def _map_tensor_name(gguf_name: str) -> str | None:
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"
    m = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
    if not m:
        return None
    idx, rest = m.group(1), m.group(2)
    mapped_rest = _BLOCK_TENSOR_MAP.get(rest)
    if mapped_rest is None:
        return None
    return f"model.layers.{idx}.{mapped_rest}"


def _read_scalar(field: gguf.ReaderField):
    value = field.parts[field.data[0]]
    if field.types[-1] == gguf.GGUFValueType.STRING:
        return bytes(value).decode("utf-8")
    return value[0].item()


def _get_module(root, dotted_path: str):
    obj = root
    for part in dotted_path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _tensor_shape(t) -> tuple[int, ...]:
    return tuple(int(d) for d in reversed(t.shape))


def load_gguf(path: str, device: str = "cuda", compute_dtype: torch.dtype = torch.bfloat16) -> Qwen2ForCausalLM:
    reader = gguf.GGUFReader(path)

    def scalar(key: str):
        return _read_scalar(reader.fields[key])

    tensors_by_name = {t.name: t for t in reader.tensors}
    embd_shape = _tensor_shape(tensors_by_name["token_embd.weight"])

    cfg = ModelConfig(
        vocab_size=embd_shape[0],
        hidden_size=scalar("qwen2.embedding_length"),
        intermediate_size=scalar("qwen2.feed_forward_length"),
        num_hidden_layers=scalar("qwen2.block_count"),
        num_attention_heads=scalar("qwen2.attention.head_count"),
        num_key_value_heads=scalar("qwen2.attention.head_count_kv"),
        rms_norm_eps=scalar("qwen2.attention.layer_norm_rms_epsilon"),
        rope_theta=scalar("qwen2.rope.freq_base"),
        max_position_embeddings=scalar("qwen2.context_length"),
        tie_word_embeddings=False,  # we always materialize an explicit lm_head below
    )

    # Meta device: constructing Qwen2ForCausalLM the normal way allocates a
    # real, full-precision (fp32, PyTorch's default) nn.Linear for every
    # attention/MLP matrix -- including the ones about to be thrown away and
    # replaced with a compact QuantizedLinear below. For a 7B+ model that
    # transient skeleton is 25-30GB of host RAM for weights that live for a
    # few milliseconds, easily exceeding a normal machine's free RAM and
    # crashing (this is exactly what happened testing a 7B checkpoint: a
    # hard segfault during construction, before any GGUF tensor was even
    # touched). Building on the meta device instead allocates shape/dtype
    # metadata only, no storage -- real data lands only where state_dict
    # (assign=True below) or the QuantizedLinear replacement loop actually
    # puts it.
    with torch.device("meta"):
        model = Qwen2ForCausalLM(cfg)
    # The only non-persistent buffer in the model -- meta construction skips
    # it entirely (persistent=False means it's never in state_dict either),
    # so it needs recomputing for real rather than relying on load_state_dict.
    model.model.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    state_dict: dict[str, torch.Tensor] = {}
    quantized: dict[str, tuple[torch.Tensor, int, tuple[int, int]]] = {}

    for name, t in tensors_by_name.items():
        mapped = _map_tensor_name(name)
        if mapped is None:
            continue

        ggml_type = int(t.tensor_type)
        # lm_head's full matrix is used on every forward call anyway (logits
        # over the whole vocab), so on-the-fly dequant wastes nothing there --
        # same as the per-layer matrices. embed_tokens is the opposite: only
        # one row per token is ever needed, so dequantizing the whole 150k-row
        # table on every call just to read one row would be wasteful; it (and
        # everything else -- norms, biases) gets dequantized once, eagerly,
        # regardless of whether the source tensor happens to be quantized.
        is_quantizable_linear = mapped == "lm_head.weight" or any(mapped.endswith(s) for s in _LINEAR_SUFFIXES)

        if is_quantizable_linear and ggml_type not in (GGML_TYPE_F32, GGML_TYPE_F16):
            raw = torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1).copy())
            quantized[mapped] = (raw, ggml_type, _tensor_shape(t))
        else:
            raw = torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1).copy())
            flat = dequantize(raw, ggml_type, t.n_elements)
            state_dict[mapped] = flat.reshape(_tensor_shape(t))

    # assign=True: the model's own parameters are meta tensors with no
    # storage, so the normal copy_-into-existing-tensor behavior can't work
    # here -- assign replaces them outright with the real tensors from
    # state_dict instead.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    missing = set(missing) - set(quantized.keys())
    if missing:
        raise RuntimeError(f"Missing keys after GGUF load: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys after GGUF load: {unexpected}")

    for weight_name, (raw, ggml_type, shape) in quantized.items():
        # weight_name is "...some.path.module_name.weight" -- the *module*
        # to replace is everything but the trailing ".weight" (e.g.
        # "model.layers.0.self_attn.q_proj" or, for lm_head, just
        # "lm_head" with no further nesting at all).
        module_path = weight_name.removesuffix(".weight")
        if "." in module_path:
            parent_path, child_name = module_path.rsplit(".", 1)
            parent = _get_module(model, parent_path)
        else:
            child_name = module_path
            parent = model

        out_features, in_features = shape
        bias = state_dict.get(f"{module_path}.bias")
        setattr(
            parent,
            child_name,
            QuantizedLinear(raw, ggml_type, out_features, in_features, bias=bias, compute_dtype=compute_dtype),
        )

    model = model.to(device=device, dtype=compute_dtype)
    model.eval()
    return model
