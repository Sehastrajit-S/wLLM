"""P1.1 baseline: thin wrapper around `transformers` used as the ground-truth
reference that later phases (native model runner, PagedAttention kernels,
scheduler) get diffed against.
"""
from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer
