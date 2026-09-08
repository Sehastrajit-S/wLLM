import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from safetensors.torch import save_file

from wllm.engine.lora import LoRARegistry, apply_lora_delta

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_registry_with_adapter(adapter_id: str, layer_idx: int, module_name: str, rank=4, in_f=8, out_f=8, scale=2.0):
    registry = LoRARegistry()
    A = torch.randn(rank, in_f, device=DEVICE)
    B = torch.randn(out_f, rank, device=DEVICE)
    registry._adapters[adapter_id] = {(layer_idx, module_name): (A, B, scale)}
    return registry, A, B


def test_apply_lora_delta_matches_manual_computation():
    registry, A, B = _make_registry_with_adapter("adapter_x", layer_idx=0, module_name="q_proj", scale=2.0)

    x = torch.randn(2, 3, 8, device=DEVICE)  # (batch=2, seq=3, in_features=8)
    base_out = torch.randn(2, 3, 8, device=DEVICE)
    lora_ids = ["adapter_x", "adapter_x"]

    out = apply_lora_delta(x, base_out, lora_ids, registry, layer_idx=0, module_name="q_proj")

    expected_delta = torch.matmul(torch.matmul(x, A.T), B.T) * 2.0
    assert torch.allclose(out, base_out + expected_delta, atol=1e-4)


def test_none_lora_ids_is_a_noop():
    registry, _, _ = _make_registry_with_adapter("adapter_x", 0, "q_proj")
    x = torch.randn(2, 3, 8, device=DEVICE)
    base_out = torch.randn(2, 3, 8, device=DEVICE)
    out = apply_lora_delta(x, base_out, None, registry, 0, "q_proj")
    assert torch.equal(out, base_out)


def test_no_registry_is_a_noop():
    x = torch.randn(2, 3, 8, device=DEVICE)
    base_out = torch.randn(2, 3, 8, device=DEVICE)
    out = apply_lora_delta(x, base_out, ["a", "a"], None, 0, "q_proj")
    assert torch.equal(out, base_out)


def test_mixed_batch_rows_isolated_no_cross_contamination():
    """Three rows: one with adapter A, one with adapter B, one with none --
    each must get exactly its own treatment.
    """
    registry = LoRARegistry()
    A1 = torch.randn(4, 8, device=DEVICE)
    B1 = torch.randn(8, 4, device=DEVICE)
    A2 = torch.randn(4, 8, device=DEVICE)
    B2 = torch.randn(8, 4, device=DEVICE)
    registry._adapters["adapter_1"] = {(0, "q_proj"): (A1, B1, 1.0)}
    registry._adapters["adapter_2"] = {(0, "q_proj"): (A2, B2, 3.0)}

    x = torch.randn(3, 2, 8, device=DEVICE)
    base_out = torch.randn(3, 2, 8, device=DEVICE)
    lora_ids = ["adapter_1", None, "adapter_2"]

    out = apply_lora_delta(x, base_out, lora_ids, registry, 0, "q_proj")

    expected_row0 = base_out[0] + torch.matmul(torch.matmul(x[0], A1.T), B1.T) * 1.0
    expected_row1 = base_out[1]  # untouched
    expected_row2 = base_out[2] + torch.matmul(torch.matmul(x[2], A2.T), B2.T) * 3.0

    assert torch.allclose(out[0], expected_row0, atol=1e-4)
    assert torch.allclose(out[1], expected_row1, atol=1e-6)
    assert torch.allclose(out[2], expected_row2, atol=1e-4)


def test_adapter_with_no_weights_for_this_module_is_a_noop():
    registry, _, _ = _make_registry_with_adapter("adapter_x", layer_idx=0, module_name="q_proj")
    x = torch.randn(1, 2, 8, device=DEVICE)
    base_out = torch.randn(1, 2, 8, device=DEVICE)
    # this adapter has no weights registered for "v_proj"
    out = apply_lora_delta(x, base_out, ["adapter_x"], registry, layer_idx=0, module_name="v_proj")
    assert torch.equal(out, base_out)


def test_load_adapter_reads_peft_format_directory(tmp_path):
    rank, in_f, out_f, alpha = 4, 8, 8, 8.0
    config = {"r": rank, "lora_alpha": alpha, "target_modules": ["q_proj"]}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))

    A = torch.randn(rank, in_f)
    B = torch.randn(out_f, rank)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B,
        },
        str(tmp_path / "adapter_model.safetensors"),
    )

    registry = LoRARegistry()
    registry.load_adapter("my_adapter", str(tmp_path), device=DEVICE, dtype=torch.float32)

    assert registry.list_adapters() == ["my_adapter"]
    loaded_A, loaded_B, scale = registry.get("my_adapter", 0, "q_proj")
    assert torch.allclose(loaded_A.cpu(), A)
    assert torch.allclose(loaded_B.cpu(), B)
    assert scale == pytest.approx(alpha / rank)


def test_unload_adapter_removes_it():
    registry, _, _ = _make_registry_with_adapter("adapter_x", 0, "q_proj")
    registry.unload_adapter("adapter_x")
    assert registry.list_adapters() == []
    assert registry.get("adapter_x", 0, "q_proj") is None
