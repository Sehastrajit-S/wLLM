"""GGUF weight dequantization. Written from the block-quantization formulas
llama.cpp/ggml use (not derived from the `gguf` package's own reference
dequantizer -- that's used only in tests to cross-validate this).

Q8_0 block (34 bytes / 32 elements): fp16 scale `d` + 32 signed int8 values.
    value[i] = qs[i] * d
Q4_0 block (18 bytes / 32 elements): fp16 scale `d` + 16 bytes of packed
4-bit values (2 per byte: low nibble = element i, high nibble = element i+16),
with a symmetric zero-point of 8 (4-bit range 0..15 -> -8..7):
    value[i] = (nibble[i] - 8) * d

Runs on whatever device `raw` lives on (including CUDA) so quantized weights
can be dequantized on the fly during the forward pass without an extra
CPU round-trip -- see quant_linear.py.
"""
from __future__ import annotations

import torch

QK8_0 = 32
BLOCK_BYTES_Q8_0 = 2 + QK8_0  # 34: fp16 scale + 32 int8 values

QK4_0 = 32
BLOCK_BYTES_Q4_0 = 2 + QK4_0 // 2  # 18: fp16 scale + 16 packed-nibble bytes

# GGML type ids (from gguf.GGMLQuantizationType) this module supports.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8


def dequantize_q8_0(raw: torch.Tensor, n_elements: int) -> torch.Tensor:
    assert raw.dtype == torch.uint8
    assert n_elements % QK8_0 == 0
    n_blocks = n_elements // QK8_0
    blocks = raw.view(n_blocks, BLOCK_BYTES_Q8_0)

    scale = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)  # (n_blocks, 1)
    qs = blocks[:, 2:].contiguous().view(torch.int8).to(torch.float32)  # (n_blocks, 32)

    return (qs * scale).reshape(-1)


def dequantize_q4_0(raw: torch.Tensor, n_elements: int) -> torch.Tensor:
    assert raw.dtype == torch.uint8
    assert n_elements % QK4_0 == 0
    n_blocks = n_elements // QK4_0
    blocks = raw.view(n_blocks, BLOCK_BYTES_Q4_0)

    scale = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)  # (n_blocks, 1)
    packed = blocks[:, 2:].contiguous()  # (n_blocks, 16) uint8

    low = (packed & 0x0F).to(torch.float32)  # elements 0..15
    high = (packed >> 4).to(torch.float32)  # elements 16..31
    nibbles = torch.cat([low, high], dim=1)  # (n_blocks, 32), correctly ordered

    return ((nibbles - 8.0) * scale).reshape(-1)


def dequantize(raw: torch.Tensor, ggml_type: int, n_elements: int) -> torch.Tensor:
    """Returns a flat float32 tensor of `n_elements` values -- caller reshapes."""
    if ggml_type == GGML_TYPE_F32:
        return raw.view(torch.float32).reshape(-1)[:n_elements].clone()
    if ggml_type == GGML_TYPE_F16:
        return raw.view(torch.float16).reshape(-1)[:n_elements].to(torch.float32)
    if ggml_type == GGML_TYPE_Q8_0:
        return dequantize_q8_0(raw, n_elements)
    if ggml_type == GGML_TYPE_Q4_0:
        return dequantize_q4_0(raw, n_elements)
    raise NotImplementedError(
        f"GGML type {ggml_type} not supported -- only F32/F16/Q8_0/Q4_0 are implemented. "
        "K-quants (Q4_K_M etc.) and I-quants have a more complex super-block structure "
        "and are deferred."
    )
