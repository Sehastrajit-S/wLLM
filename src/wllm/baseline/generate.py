from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.inference_mode()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    max_new_tokens: int = 64,
    greedy: bool = True,
) -> str:
    """Greedy by default so output is deterministic and diffable across runs/phases."""
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=None if greedy else 0.7,
        top_p=None if greedy else 0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.inference_mode()
def capture_reference_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
) -> dict[str, torch.Tensor]:
    """Runs a single forward pass (no generation) over the prompt and returns the
    input_ids + full logits tensor. This is the artifact P1.2's PagedAttention
    kernel output gets diffed against for correctness.
    """
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    outputs = model(**inputs)
    return {
        "input_ids": inputs["input_ids"].cpu(),
        "logits": outputs.logits.float().cpu(),
    }
