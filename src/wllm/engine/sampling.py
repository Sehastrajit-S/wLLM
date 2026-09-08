"""Batched sampler: temperature, top-k, top-p (nucleus), min-p, and
repetition penalty, applied per-row so sequences in the same batch can carry
different sampling params. temperature == 0.0 means greedy for that row --
deterministic argmax on the repetition-penalized logits, bypassing every
stochastic filter (this is what every earlier phase's correctness tests rely
on: SamplingParams() defaults to greedy so nothing here changes their
behavior unless a sequence opts into sampling).

Beam search is NOT implemented -- it needs per-beam KV cache duplication,
which doesn't fit the current one-cache-slot-per-sequence model without
real surgery on the scheduler/cache manager. Deferred, same as chunked
prefill and prefix caching before it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0.0 == greedy
    top_k: int = 0  # 0 == disabled
    top_p: float = 1.0  # 1.0 == disabled
    min_p: float = 0.0  # 0.0 == disabled
    repetition_penalty: float = 1.0  # 1.0 == disabled
    seed: int | None = None


def apply_token_mask(logits: torch.Tensor, masks: list[torch.Tensor | None]) -> torch.Tensor:
    """masks[i]: (vocab,) bool tensor of ALLOWED tokens for row i (e.g. from
    a JSONSchemaGuide), or None for an unconstrained row. Applied first,
    before anything else -- a grammar-disallowed token must never be
    selectable regardless of repetition penalty, temperature, or top-k/p.
    """
    logits = logits.clone()
    for i, mask in enumerate(masks):
        if mask is not None:
            logits[i] = logits[i].masked_fill(~mask, float("-inf"))
    return logits


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[list[int]], penalties: list[float]) -> torch.Tensor:
    """HF-style penalty: divide positive logits / multiply negative logits of
    already-generated tokens by `penalty`, per row. Loops over rows in Python
    since each sequence's history is a different length (ragged) -- fine at
    engine-loop batch sizes.
    """
    logits = logits.clone()
    for i, (ids, penalty) in enumerate(zip(generated_ids, penalties)):
        if penalty == 1.0 or not ids:
            continue
        idx = torch.tensor(sorted(set(ids)), device=logits.device, dtype=torch.long)
        row = logits[i, idx]
        logits[i, idx] = torch.where(row > 0, row / penalty, row * penalty)
    return logits


def _filter_top_k_top_p_min_p(
    logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor, min_p: torch.Tensor
) -> torch.Tensor:
    """logits: (batch, vocab). top_k/top_p/min_p: (batch,) -- per-row params,
    so different sequences in the same batch can use different values.
    Returns logits with disallowed positions set to -inf.
    """
    batch, vocab = logits.shape
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)

    keep = torch.ones_like(sorted_logits, dtype=torch.bool)

    # top-k: rank < k (k == 0 means "disabled" -> keep everything)
    rank = torch.arange(vocab, device=logits.device).unsqueeze(0).expand(batch, -1)
    k_active = top_k > 0
    keep &= ~k_active.unsqueeze(-1) | (rank < top_k.clamp(min=1).unsqueeze(-1))

    # top-p (nucleus): smallest prefix whose cumulative prob >= p. Always
    # keep at least the top-1 token even if it alone exceeds p.
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)
    p_active = top_p < 1.0
    over_p = cumprobs > top_p.unsqueeze(-1)
    over_p[:, 0] = False  # never drop the top token
    keep &= ~p_active.unsqueeze(-1) | ~over_p

    # min-p: drop tokens with prob < min_p * max_prob
    max_prob = probs[:, :1]
    mp_active = min_p > 0.0
    below_min_p = probs < (min_p.unsqueeze(-1) * max_prob)
    below_min_p[:, 0] = False
    keep &= ~mp_active.unsqueeze(-1) | ~below_min_p

    sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))

    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


@torch.inference_mode()
def sample(
    logits: torch.Tensor,
    params: list[SamplingParams],
    generated_ids: list[list[int]],
    generator: torch.Generator | None = None,
    token_masks: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """logits: (batch, vocab) for the current step only.
    Returns (token_ids: (batch,), logprobs: (batch,)) -- logprob is under the
    temperature-scaled, repetition-penalized distribution (post those two
    deterministic transforms, pre top-k/p/min-p truncation), so it reflects
    the model's adjusted confidence rather than a truncated distribution.
    """
    logits = logits.float()
    if token_masks is not None:
        logits = apply_token_mask(logits, token_masks)
    penalties = [p.repetition_penalty for p in params]
    logits = apply_repetition_penalty(logits, generated_ids, penalties)

    is_greedy = torch.tensor([p.temperature == 0.0 for p in params], device=logits.device)
    temperature = torch.tensor([max(p.temperature, 1e-5) for p in params], device=logits.device)
    scaled = logits / temperature.unsqueeze(-1)

    # Greedy rows report logprob under the raw (temperature=1) distribution --
    # their actual sampling temperature is clamped near zero purely to keep
    # `scaled` finite for the (unused, since argmax wins below) filter/sample
    # path, which would otherwise collapse log_softmax to a degenerate ~0.
    logprob_temperature = torch.where(is_greedy, torch.ones_like(temperature), temperature)
    logprobs_full = F.log_softmax(logits / logprob_temperature.unsqueeze(-1), dim=-1)
    greedy_tokens = logits.argmax(dim=-1)

    top_k = torch.tensor([p.top_k for p in params], device=logits.device)
    top_p = torch.tensor([p.top_p for p in params], device=logits.device)
    min_p = torch.tensor([p.min_p for p in params], device=logits.device)
    filtered = _filter_top_k_top_p_min_p(scaled, top_k, top_p, min_p)
    probs = F.softmax(filtered, dim=-1)

    sampled_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    token_ids = torch.where(is_greedy, greedy_tokens, sampled_tokens)
    token_logprobs = logprobs_full.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_ids, token_logprobs
