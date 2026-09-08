"""GGUF weight dequantization. Written from the block-quantization formulas
llama.cpp/ggml use (not derived from the `gguf` package's own reference
dequantizer -- that's used only in tests to cross-validate this).

Q8_0 block (34 bytes / 32 elements): fp16 scale `d` + 32 signed int8 values.
    value[i] = qs[i] * d
Q4_0 block (18 bytes / 32 elements): fp16 scale `d` + 16 bytes of packed
4-bit values (2 per byte: low nibble = element i, high nibble = element i+16),
with a symmetric zero-point of 8 (4-bit range 0..15 -> -8..7):
    value[i] = (nibble[i] - 8) * d
Q4_1 block (20 bytes / 32 elements): fp16 scale `d` + fp16 min `m` + the same
16-byte nibble packing as Q4_0, but with an affine (not symmetric) zero
point -- the raw 0..15 nibble is used directly, offset by `m` instead of a
fixed -8:
    value[i] = nibble[i] * d + m
Real-world "legacy" GGUF quant files (e.g. anything named Q4_0/Q5_0/Q8_0)
routinely mix in a handful of Q4_1 tensors for specific layers llama.cpp's
own quantizer considers more sensitive (observed: the first few layers'
ffn_down.weight) -- supporting only Q4_0 leaves those files unreadable.

Q6_K block (210 bytes / 256 elements, a "K-quant" super-block -- more
involved than the legacy types above): 128 bytes `ql` (low 4 bits of each
6-bit value, 2 values/byte), 64 bytes `qh` (high 2 bits of each value, 4
values/byte), 16 signed-int8 sub-block scales (one per 16-element sub-block),
and one fp16 super-block scale `d`. Value i in sub-block `is = i//16`
(0..15) reconstructs its 6-bit quantized value from the matching `ql`/`qh`
bits (assembled into a 0..63 range, then recentered by -32 to signed
-32..31), scaled by `d * scales[is]`. The exact bit layout follows llama.cpp's
`dequantize_row_q6_K` (processed in two 128-element halves per 256-element
block; see dequantize_q6_k's implementation for the index arithmetic).
Real-world "legacy"-named GGUF files almost universally use Q6_K for
output.weight/token_embd.weight specifically (llama.cpp's quantizer upgrades
those two tensors for quality regardless of the file's nominal quant level)
-- supporting only the legacy types leaves effectively every real GGUF file
unreadable at the output layer. Other K-quants (Q4_K/Q5_K/etc.) and I-quants
remain unsupported; add them the same way if a real file needs one.

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

QK4_1 = 32
BLOCK_BYTES_Q4_1 = 2 + 2 + QK4_1 // 2  # 20: fp16 scale + fp16 min + 16 packed-nibble bytes

QK_K = 256
BLOCK_BYTES_Q6_K = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2  # 210: ql(128) + qh(64) + scales(16) + fp16 d

# GGML type ids (from gguf.GGMLQuantizationType) this module supports.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q6_K = 14


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


def dequantize_q4_1(raw: torch.Tensor, n_elements: int) -> torch.Tensor:
    assert raw.dtype == torch.uint8
    assert n_elements % QK4_1 == 0
    n_blocks = n_elements // QK4_1
    blocks = raw.view(n_blocks, BLOCK_BYTES_Q4_1)

    scale = blocks[:, 0:2].contiguous().view(torch.float16).to(torch.float32)  # (n_blocks, 1)
    minimum = blocks[:, 2:4].contiguous().view(torch.float16).to(torch.float32)  # (n_blocks, 1)
    packed = blocks[:, 4:].contiguous()  # (n_blocks, 16) uint8

    low = (packed & 0x0F).to(torch.float32)  # elements 0..15
    high = (packed >> 4).to(torch.float32)  # elements 16..31
    nibbles = torch.cat([low, high], dim=1)  # (n_blocks, 32), correctly ordered

    return (nibbles * scale + minimum).reshape(-1)


def _dequantize_q6_k_half(ql: torch.Tensor, qh: torch.Tensor, scales: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """One 128-element half of a Q6_K super-block. ql: (n,64), qh: (n,32),
    scales: (n,8) -- all already int32/float32 -- d: (n,1) float32.
    """
    ql = ql.to(torch.int32)
    qh = qh.to(torch.int32)
    ql_lo, ql_hi = ql[:, :32], ql[:, 32:]  # ql[l+0], ql[l+32] for l in 0..31

    q1 = ((ql_lo & 0xF) | (((qh >> 0) & 3) << 4)) - 32
    q2 = ((ql_hi & 0xF) | (((qh >> 2) & 3) << 4)) - 32
    q3 = ((ql_lo >> 4) | (((qh >> 4) & 3) << 4)) - 32
    q4 = ((ql_hi >> 4) | (((qh >> 6) & 3) << 4)) - 32

    # is = l // 16 in {0, 1} for l in 0..31 -- each of the two sub-block
    # scales for a given (q1..q4) applies to 16 consecutive l's.
    sc1 = scales[:, 0:2].repeat_interleave(16, dim=1)
    sc2 = scales[:, 2:4].repeat_interleave(16, dim=1)
    sc3 = scales[:, 4:6].repeat_interleave(16, dim=1)
    sc4 = scales[:, 6:8].repeat_interleave(16, dim=1)

    y1 = d * sc1 * q1.to(torch.float32)
    y2 = d * sc2 * q2.to(torch.float32)
    y3 = d * sc3 * q3.to(torch.float32)
    y4 = d * sc4 * q4.to(torch.float32)
    return torch.cat([y1, y2, y3, y4], dim=1)  # (n, 128), positions l+0/32/64/96


def dequantize_q6_k(raw: torch.Tensor, n_elements: int) -> torch.Tensor:
    assert raw.dtype == torch.uint8
    assert n_elements % QK_K == 0
    n_blocks = n_elements // QK_K
    blocks = raw.view(n_blocks, BLOCK_BYTES_Q6_K)

    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].contiguous().view(torch.int8).to(torch.float32)  # (n_blocks, 16)
    d = blocks[:, 208:210].contiguous().view(torch.float16).to(torch.float32)  # (n_blocks, 1)

    first = _dequantize_q6_k_half(ql[:, :64], qh[:, :32], scales[:, :8], d)
    second = _dequantize_q6_k_half(ql[:, 64:], qh[:, 32:], scales[:, 8:], d)
    return torch.cat([first, second], dim=1).reshape(-1)


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
    if ggml_type == GGML_TYPE_Q4_1:
        return dequantize_q4_1(raw, n_elements)
    if ggml_type == GGML_TYPE_Q6_K:
        return dequantize_q6_k(raw, n_elements)
    raise NotImplementedError(
        f"GGML type {ggml_type} not supported -- only F32/F16/Q8_0/Q4_0/Q4_1/Q6_K are implemented. "
        "Other K-quants (Q4_K/Q5_K/etc.) and I-quants have their own super-block structures "
        "and are deferred."
    )
