"""Multi-LoRA serving: different requests batched together can each use a
different adapter (or none) simultaneously. LoRA modifies a linear layer as
    y = x @ W.T + b               (base, frozen)
    y_lora = y + scale * (x @ A.T) @ B.T
with A: (rank, in_features), B: (out_features, rank), scale = alpha / rank
(the standard PEFT convention -- rank-stabilized variants aren't handled).

Real vLLM applies this via custom batched (punica-style) kernels that do the
whole batch's heterogeneous-adapter delta in one launch. Correctness-first
here instead: group the batch by which adapter each row uses, and for each
present adapter, gather its rows, compute that group's delta, and scatter it
back onto the base projection's output. One extra pair of small matmuls per
adapter present in a step rather than per row, and a plain nn.Linear stays a
plain nn.Linear -- no wrapping/replacing modules, adapters are looked up by
(layer_idx, module_name) from a registry instead.

Loading reads the standard PEFT adapter format (adapter_config.json +
adapter_model.safetensors) -- that's the part with no engineering value in
reimplementing differently; the actual serving-time application above is
the part that's ours.
"""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file


def apply_lora_delta(
    x: torch.Tensor,
    base_out: torch.Tensor,
    lora_ids: list[str | None] | None,
    registry: LoRARegistry | None,
    layer_idx: int,
    module_name: str,
) -> torch.Tensor:
    """x: (batch, seq, in_features), base_out: (batch, seq, out_features) --
    same batch dim, one lora_id per row. Returns base_out with each row's
    adapter delta (if any) added; unchanged if lora_ids/registry aren't given
    or no row in this batch has an adapter touching this module.
    """
    if not lora_ids or registry is None:
        return base_out

    unique_ids = {lid for lid in lora_ids if lid is not None}
    if not unique_ids:
        return base_out

    for adapter_id in unique_ids:
        weights = registry.get(adapter_id, layer_idx, module_name)
        if weights is None:
            continue
        A, B, scale = weights
        row_idx = torch.tensor([i for i, lid in enumerate(lora_ids) if lid == adapter_id], device=x.device)
        x_rows = x.index_select(0, row_idx).to(A.dtype)
        delta = torch.matmul(torch.matmul(x_rows, A.transpose(0, 1)), B.transpose(0, 1)) * scale
        base_out = base_out.index_add(0, row_idx, delta.to(base_out.dtype))

    return base_out


class LoRARegistry:
    def __init__(self):
        self._adapters: dict[str, dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, float]]] = {}

    def load_adapter(self, adapter_id: str, path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        with open(os.path.join(path, "adapter_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        rank = config["r"]
        scale = config["lora_alpha"] / rank

        state_dict = {}
        weights_path = os.path.join(path, "adapter_model.safetensors")
        state_dict.update(load_file(weights_path))

        weights: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, float]] = {}
        for key, tensor in state_dict.items():
            if not key.endswith("lora_A.weight"):
                continue
            b_key = key.replace("lora_A.weight", "lora_B.weight")
            if b_key not in state_dict:
                continue
            layer_idx, module_name = _parse_lora_key(key)
            if layer_idx is None:
                continue
            A = tensor.to(device=device, dtype=dtype)
            B = state_dict[b_key].to(device=device, dtype=dtype)
            weights[(layer_idx, module_name)] = (A, B, scale)

        self._adapters[adapter_id] = weights

    def unload_adapter(self, adapter_id: str) -> None:
        self._adapters.pop(adapter_id, None)

    def list_adapters(self) -> list[str]:
        return list(self._adapters.keys())

    def get(self, adapter_id: str, layer_idx: int, module_name: str):
        return self._adapters.get(adapter_id, {}).get((layer_idx, module_name))


def _parse_lora_key(key: str) -> tuple[int | None, str | None]:
    """PEFT's saved key looks like:
    "...model.layers.{i}.self_attn.{module}.lora_A.weight" or
    "...model.layers.{i}.mlp.{module}.lora_A.weight"
    -- the leading prefix (base_model.model. etc.) varies by how the base
    model was wrapped, so we just look for the "layers.{i}...{module}."
    pattern rather than anchoring to a specific prefix.
    """
    parts = key.split(".")
    try:
        layers_idx = parts.index("layers")
        layer_num = int(parts[layers_idx + 1])
        module_name = parts[layers_idx + 3]  # .../layers/{i}/{self_attn|mlp}/{module}/lora_A/weight
        return layer_num, module_name
    except (ValueError, IndexError):
        return None, None
