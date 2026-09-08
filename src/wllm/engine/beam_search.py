"""Beam search: unlike every other generation mode in this codebase, a
single logical request explores `num_beams` candidate continuations at once
and keeps only the `num_beams` highest length-penalized cumulative-logprob
paths after each step -- so which token history survives to the next step
isn't fixed the way it is for Sequence-based generation (a parent beam can
spawn zero, one, or more than one surviving child, or die out entirely).

Because of that branching, a "beam" here isn't tied to one persistent
Sequence/KV-cache identity across steps the way regular decoding is. Instead,
each step re-derives every active beam's next-token logits from scratch via
prefill_with_cache_and_prefix_reuse(cache, scratch_seq_id, full_token_history)
-- the exact same prefix-caching-accelerated primitive prefix caching and
chunked prefill already use, then immediately frees that scratch cache entry.
With enable_prefix_caching=True on the cache, this costs at most one block's
worth of real recompute per beam per step (everything before the trailing
partial block hash-matches an earlier step's freed-but-still-hash-indexed
blocks and is reused, never recomputed); with prefix caching off, it degrades
to full O(length) recompute per beam per step -- correct either way, just
slower without it. This trades a little redundant compute for reusing every
existing, already-tested cache primitive instead of teaching the KV cache
manager a new kind of copy-on-write beam-forking operation.

Real vLLM instead forks the KV cache itself (block-level copy-on-write) so a
continuing beam never repeats work its parent already paid for, and batches
all beams of a step into one forward pass. Neither is done here -- beams are
processed one at a time per step (same tradeoff speculative decoding already
makes in this codebase, see scheduler.py) -- documented tradeoffs, not
oversights.

Stopping rule is the standard simplified one: once `num_beams` candidates
have reached EOS/a stop token (enough to fill every beam slot with a
finished candidate) or every active beam has generated max_new_tokens, the
group is done and its single highest-scoring finished (or, failing that,
active) beam is the result. This is not HF's more elaborate early-stopping
heuristic (which can also stop earlier by proving no future active beam
could ever outscore the worst already-finished one) -- simpler, and still
correct, just not always maximally early.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BeamSearchParams:
    num_beams: int
    length_penalty: float = 1.0


@dataclass
class _Beam:
    token_ids: list[int]  # prompt + generated so far
    cum_logprob: float


class BeamGroup:
    """One logical beam-search request. `group_id` is opaque bookkeeping the
    Scheduler never inspects -- AsyncEngine uses it to match a completed
    group back to the caller awaiting it.
    """

    def __init__(
        self,
        group_id: int,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        params: BeamSearchParams,
        eos_token_id: int | None = None,
        stop_token_ids: frozenset[int] = frozenset(),
        lora_id: str | None = None,
    ):
        self.group_id = group_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.prompt_len = len(prompt_token_ids)
        self.max_new_tokens = max_new_tokens
        self.params = params
        self.eos_token_id = eos_token_id
        self.stop_token_ids = stop_token_ids
        self.lora_id = lora_id

        self.beams: list[_Beam] = [_Beam(token_ids=list(prompt_token_ids), cum_logprob=0.0)]
        self.finished: list[_Beam] = []
        self.result: list[int] | None = None  # winning beam's generated (post-prompt) token ids, once done
        self.finish_reason: str | None = None  # "stop" (winner reached EOS/a stop token) or "length"

    def is_done(self) -> bool:
        return self.result is not None

    def _score(self, beam: _Beam) -> float:
        length = max(len(beam.token_ids) - self.prompt_len, 1)
        return beam.cum_logprob / (length ** self.params.length_penalty)

    def _is_stop(self, token_id: int) -> bool:
        return token_id == self.eos_token_id or token_id in self.stop_token_ids

    def _finalize(self) -> None:
        pool = self.finished if self.finished else self.beams
        best = max(pool, key=self._score)
        self.result = best.token_ids[self.prompt_len :]
        self.finish_reason = "stop" if self.finished else "length"
        self.beams = []

    @torch.inference_mode()
    def step(self, model, cache, lora_registry=None) -> None:
        """Advances every active beam by exactly one token. Call repeatedly
        (typically once per Scheduler.step()) until is_done() is True.
        """
        if self.is_done():
            return

        num_beams = self.params.num_beams
        candidates: list[tuple[float, int, int]] = []  # (score, parent_beam_idx, token_id)

        for i, beam in enumerate(self.beams):
            # Negative, group-namespaced scratch ids can never collide with
            # the Scheduler's own (non-negative, itertools.count()) seq_ids,
            # and are reused fresh (create -> use -> free) every step/beam --
            # nothing here persists across calls.
            scratch_id = -(self.group_id * 10_000 + i + 1)
            logits = model.prefill_with_cache_and_prefix_reuse(
                cache, scratch_id, beam.token_ids, lora_id=self.lora_id, lora_registry=lora_registry
            )
            cache.free_sequence(scratch_id)

            log_probs = logits[0, -1, :].float().log_softmax(dim=-1)
            # Only a beam's own top-`num_beams` tokens can ever land in the
            # global top `num_beams`: any candidate outside that set is
            # strictly dominated by candidates already in it (same parent
            # score, higher per-token logprob) -- the standard beam-search
            # pruning shortcut, avoids scoring the full vocab per beam pair.
            topk_logprobs, topk_tokens = log_probs.topk(num_beams)
            for logprob, token_id in zip(topk_logprobs.tolist(), topk_tokens.tolist()):
                candidates.append((beam.cum_logprob + logprob, i, token_id))

        candidates.sort(key=lambda c: c[0], reverse=True)

        new_beams: list[_Beam] = []
        for score, parent_idx, token_id in candidates:
            if len(new_beams) >= num_beams:
                break
            parent = self.beams[parent_idx]
            child = _Beam(token_ids=parent.token_ids + [token_id], cum_logprob=score)
            if self._is_stop(token_id):
                self.finished.append(child)
            else:
                new_beams.append(child)

        self.beams = new_beams
        reached_max_len = bool(self.beams) and (len(self.beams[0].token_ids) - self.prompt_len >= self.max_new_tokens)

        if not self.beams or len(self.finished) >= num_beams or reached_max_len:
            self._finalize()
