import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gguf
import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download

from wllm.quant.dequant import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    dequantize,
)

REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"


@pytest.fixture(scope="module")
def q8_0_path():
    return hf_hub_download(REPO, "qwen2.5-0.5b-instruct-q8_0.gguf")


@pytest.fixture(scope="module")
def q4_0_path():
    return hf_hub_download(REPO, "qwen2.5-0.5b-instruct-q4_0.gguf")


def _first_tensor_of_type(reader: gguf.GGUFReader, ggml_type: int):
    for t in reader.tensors:
        if t.tensor_type == ggml_type:
            return t
    raise AssertionError(f"no tensor of type {ggml_type} found")


def test_q8_0_matches_reference_dequantizer(q8_0_path):
    reader = gguf.GGUFReader(q8_0_path)
    t = _first_tensor_of_type(reader, GGML_TYPE_Q8_0)

    raw = torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1))
    mine = dequantize(raw, GGML_TYPE_Q8_0, t.n_elements)

    reference = gguf.quants.dequantize(t.data, gguf.GGMLQuantizationType(t.tensor_type)).reshape(-1)
    reference = torch.from_numpy(np.ascontiguousarray(reference))

    assert torch.equal(mine, reference), "Q8_0 dequant does not exactly match the reference implementation"


def test_q4_0_matches_reference_dequantizer(q4_0_path):
    reader = gguf.GGUFReader(q4_0_path)
    t = _first_tensor_of_type(reader, GGML_TYPE_Q4_0)

    raw = torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1))
    mine = dequantize(raw, GGML_TYPE_Q4_0, t.n_elements)

    reference = gguf.quants.dequantize(t.data, gguf.GGMLQuantizationType(t.tensor_type)).reshape(-1)
    reference = torch.from_numpy(np.ascontiguousarray(reference))

    assert torch.equal(mine, reference), "Q4_0 dequant does not exactly match the reference implementation"


def test_f32_tensor_passthrough_matches_original(q8_0_path):
    reader = gguf.GGUFReader(q8_0_path)
    t = _first_tensor_of_type(reader, GGML_TYPE_F32)

    raw = torch.from_numpy(np.ascontiguousarray(t.data).view(np.uint8).reshape(-1))
    mine = dequantize(raw, GGML_TYPE_F32, t.n_elements)

    expected = torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1))
    assert torch.equal(mine, expected)


def test_q4_0_close_to_q8_0_for_same_gguf_checkpoint(q8_0_path, q4_0_path):
    """Cross-checking dequantized weights against a *separately downloaded*
    safetensors checkpoint turned out to be comparing against a different
    model revision (even the tiny, unpermutable attn_norm vector -- which
    has nothing to do with quantization or layout -- showed the same
    inconsistent ~2x-scale mismatch), so it wasn't a valid correctness
    signal. Q8_0 and Q4_0 here come from the *same* GGUF repo/conversion,
    so they should agree closely with each other -- Q8_0 is near-lossless,
    so this checks Q4_0's larger-but-still-small quantization error against
    it instead.
    """
    reader_q8 = gguf.GGUFReader(q8_0_path)
    reader_q4 = gguf.GGUFReader(q4_0_path)
    t8 = next(t for t in reader_q8.tensors if t.name == "blk.0.attn_q.weight")
    t4 = next(t for t in reader_q4.tensors if t.name == "blk.0.attn_q.weight")

    raw8 = torch.from_numpy(np.ascontiguousarray(t8.data).reshape(-1))
    raw4 = torch.from_numpy(np.ascontiguousarray(t4.data).reshape(-1))
    dq8 = dequantize(raw8, GGML_TYPE_Q8_0, t8.n_elements)
    dq4 = dequantize(raw4, GGML_TYPE_Q4_0, t4.n_elements)

    rel_err = (dq4 - dq8).abs().mean() / dq8.abs().mean()
    print(f"\nQ4_0 vs Q8_0 mean relative error (same checkpoint) = {rel_err.item():.4f}")
    assert rel_err < 0.2
