"""Continuous batching scheduler: interleaves prefill of newly-admitted
requests with a single batched decode step across every currently-running
sequence. Admission control tracks a conservative "pledged blocks" estimate
(worst-case future growth of already-running sequences) so it doesn't
over-admit relative to real KV cache capacity; preemption (recompute
strategy: drop the victim's cache and requeue it) is kept as a backstop for
cases the pledge accounting doesn't (or deliberately isn't asked to) cover.

Chunked prefill (opt-in via `max_prefill_tokens_per_step`): a long prompt is
split across multiple scheduler steps instead of processed in one shot, so
it can't stall other sequences' decode progress for an unbounded amount of
time. Built on the same "continue_prefill" primitive prefix caching uses --
each chunk attends to whatever's already cached for that sequence (a reused
prefix, earlier chunks, or both) as fixed context, causal only within the
new chunk. A sequence spends zero or more steps in SeqStatus.PREFILLING
before transitioning to RUNNING once its whole prompt is in the cache and it
gets its first sampled token. Left as a known limitation: chunked
continuation doesn't have its own preemption safety net the way decode
does -- it relies on admission control's conservative accounting to ensure
a chunk's cache.reserve() call never actually hits "out of memory" for an
already-admitted sequence.

CPU-swap eviction (opt-in via `enable_cpu_swap`): instead of the recompute
preemption strategy (drop everything, restart the sequence from scratch
later), the victim's entire KV cache is copied to host RAM and its GPU
blocks freed; when capacity allows, it's copied back and resumes decoding
exactly where it left off, generated tokens and all -- no lost work, just a
memory copy instead of a full recompute. Swapped-out sequences live in their
own `swapped` list (priority over the `waiting` queue for capacity, since
they're already in progress) and get first refusal on freed capacity each
step, before new admissions.

Speculative decoding (opt-in via `enable_speculative_decoding`): n-gram
(prompt-lookup) drafting -- no draft model to load or orchestrate, just a
search over the sequence's own token history for a previous occurrence of
its current tail (see speculative.py). When a draft of K tokens is found,
verification reuses continue_prefill (feeding [last_real_token,
*draft_tokens]) to get K+1 predictions in one forward pass, accepting the
longest matching prefix and rolling back the cache for whatever's rejected.
Restricted to greedy sequences (temperature == 0) with repetition_penalty
== 1.0 and no active guide -- that's what makes "compare draft token against
logits.argmax()" an *exact* stand-in for what sequential decode would have
produced token-by-token, rather than an approximation (repetition penalty is
history-dependent in a way that batched multi-position verification can't
cheaply replicate exactly, and stochastic sampling would need a full
rejection-sampling scheme to preserve the target distribution -- neither is
implemented here). Processed one sequence at a time rather than batched
together (each may have a different draft length or none at all) -- real
GPU-batching efficiency for heterogeneous-length speculative verification is
a further optimization, not implemented in this pass.

Beam search (`add_beam_request`): a separate scheduling lane entirely, since
a beam-search request doesn't map onto one persistent Sequence the way
everything else here does (see beam_search.py's module docstring for why and
how). Every Scheduler.step() also advances every active BeamGroup by one
token each; unlike the rest of this file, that lane has no admission control
or preemption -- see _step_beam_groups.
"""
from __future__ import annotations

import torch

from wllm.engine.beam_search import BeamGroup
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.sampling import sample
from wllm.engine.sequence import SeqStatus, Sequence
from wllm.engine.speculative import propose_ngram_draft


class Scheduler:
    def __init__(
        self,
        model,
        cache: KVCacheManager,
        device: str | torch.device | None = None,
        graph_decoder=None,
        max_prefill_tokens_per_step: int | None = None,
        enable_cpu_swap: bool = False,
        enable_speculative_decoding: bool = False,
        speculative_ngram_size: int = 3,
        speculative_max_draft_len: int = 4,
        lora_registry=None,
    ):
        """`graph_decoder`, if given, is a CUDAGraphDecoder already captured
        against `cache` -- decode steps replay its bucketed CUDA graphs
        instead of the eager per-op path (see cuda_graph_decoder.py for why
        that matters: ~3.5-8x throughput depending on batch size, since the
        eager path pays kernel-launch overhead for ~880 tiny CUDA kernels per
        step). Capacity planning (bucket sizes, scratch-sequence headroom in
        `cache`) is the caller's responsibility -- the Scheduler just uses
        whatever's handed to it. CUDAGraphDecoder already falls back to the
        eager path itself for batches/contexts too large for what was
        captured, so nothing extra is needed here for that case.

        `max_prefill_tokens_per_step`, if given, enables chunked prefill: at
        most this many prompt tokens (summed across admission + continuing
        already-in-flight chunked sequences) are processed per step, so one
        long prompt can't stall other sequences' decode progress for more
        than a step. Left None (default), a newly admitted sequence's whole
        prompt is still prefilled in one shot, exactly as before.

        `enable_cpu_swap`, if True, makes preemption use swap-to-host instead
        of the recompute strategy -- see module docstring.

        `enable_speculative_decoding`, if True, tries n-gram-drafted
        multi-token verification for eligible sequences each step (see
        module docstring for the eligibility rules and why). `..._ngram_size`
        is how many trailing tokens must match an earlier occurrence to
        trigger a draft; `..._max_draft_len` caps how many tokens a draft
        proposes at once.

        `lora_registry`, if given, is a LoRARegistry whose loaded adapters
        become selectable per-request via Sequence.lora_id. A sequence with
        lora_id set always decodes on the eager path even when a
        graph_decoder is configured -- CUDA graph capture bakes in a fixed
        computation at capture time, and per-row adapter selection varying
        between replays doesn't fit that without its own dedicated
        static-buffer treatment (not implemented here). Sequences with no
        lora_id are unaffected and still use the graph decoder as before.
        """
        self.model = model
        self.cache = cache
        # Defaulting to "cuda" unconditionally used to be a real footgun for
        # CPU inference: a CPU model + CPU cache handed to a Scheduler that
        # silently assumed "cuda" would build its own device-mismatched
        # tensors (e.g. the eager decode token_ids at the bottom of this
        # file) and crash. Deriving from the model's own parameters instead
        # means a caller who already built everything on one device never
        # needs to also remember to repeat that choice here.
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.graph_decoder = graph_decoder
        self.max_prefill_tokens_per_step = max_prefill_tokens_per_step
        self.enable_cpu_swap = enable_cpu_swap
        self.enable_speculative_decoding = enable_speculative_decoding
        self.speculative_ngram_size = speculative_ngram_size
        self.speculative_max_draft_len = speculative_max_draft_len
        self.lora_registry = lora_registry
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.swapped: list[Sequence] = []
        # Beam-search requests run on their own lane, entirely separate from
        # waiting/running/swapped -- see beam_search.py's module docstring
        # for why a "beam" doesn't fit the persistent-Sequence model the
        # rest of this file is built around. completed_beam_groups is a
        # drain queue: callers (AsyncEngine) pop from it after each step().
        self.beam_groups: list[BeamGroup] = []
        self.completed_beam_groups: list[BeamGroup] = []

    def add_request(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def add_beam_request(self, group: BeamGroup) -> None:
        self.beam_groups.append(group)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running or self.swapped or self.beam_groups)

    def cancel(self, seq_id: int) -> None:
        """Stops a sequence before it finishes naturally (client disconnect,
        stop-string match at the API layer) and frees its cache. Safe to
        call for an already-finished/removed seq_id (no-op).
        """
        for i, seq in enumerate(self.running):
            if seq.seq_id == seq_id:
                self.cache.free_sequence(seq_id)
                self.running.pop(i)
                return
        for i, seq in enumerate(self.swapped):
            if seq.seq_id == seq_id:
                self.cache.discard_swapped(seq_id)
                self.swapped.pop(i)
                return
        for i, seq in enumerate(self.waiting):
            if seq.seq_id == seq_id:
                self.waiting.pop(i)
                return

    def _blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.cache.block_size - 1) // self.cache.block_size

    def _pledged_blocks(self) -> int:
        """Worst-case additional blocks already-running sequences could still
        consume before finishing, beyond what they've used so far.
        """
        total = 0
        for seq in self.running:
            used = self._blocks_needed(seq.num_prompt_tokens + len(seq.output_token_ids))
            worst_case = self._blocks_needed(seq.total_len_estimate)
            total += max(0, worst_case - used)
        return total

    def _admit(self) -> list[Sequence]:
        admitted = []
        i = 0
        while i < len(self.waiting):
            seq = self.waiting[i]
            needed = self._blocks_needed(seq.total_len_estimate)
            available = self.cache.allocator.num_free - self._pledged_blocks()
            if needed > available:
                break  # FIFO: don't starve the head-of-line request by skipping ahead
            self._prefill(seq)
            self.running.append(seq)
            self.waiting.pop(i)
            admitted.append(seq)
        return admitted

    def _step_prefill_work(self) -> list[Sequence]:
        """Unchunked path (max_prefill_tokens_per_step is None): identical to
        the original _admit(). Chunked path: continues any sequences already
        mid-prefill, then admits new ones into whatever budget remains.
        Returns sequences whose prefill *completed* this step (got their
        first sampled token) -- a sequence still mid-chunk contributes no
        token this step and isn't included.
        """
        if self.max_prefill_tokens_per_step is None:
            return self._admit()

        budget = self.max_prefill_tokens_per_step
        completed = []

        for seq in [s for s in self.running if s.status == SeqStatus.PREFILLING]:
            if budget <= 0:
                break
            budget, done = self._continue_prefill_chunk(seq, budget)
            if done:
                completed.append(seq)

        i = 0
        while i < len(self.waiting) and budget > 0:
            seq = self.waiting[i]
            needed = self._blocks_needed(seq.total_len_estimate)
            available = self.cache.allocator.num_free - self._pledged_blocks()
            if needed > available:
                break  # FIFO: don't starve the head-of-line request by skipping ahead
            self.waiting.pop(i)
            self.running.append(seq)
            seq.status = SeqStatus.PREFILLING

            matched_blocks, _ = self.cache.match_prefix(seq.prompt_token_ids)
            if matched_blocks:
                self.cache.create_sequence_from_prefix(seq.seq_id, matched_blocks)
            else:
                self.cache.create_sequence(seq.seq_id)

            budget, done = self._continue_prefill_chunk(seq, budget)
            if done:
                completed.append(seq)

        return completed

    def _continue_prefill_chunk(self, seq: Sequence, budget: int) -> tuple[int, bool]:
        """Processes up to `budget` more of seq's prompt tokens (cache.reserve
        etc. already assumes cache.create_sequence[_from_prefix] was called).
        Returns (remaining_budget, prefill_completed_this_call).
        """
        start = self.cache.context_len(seq.seq_id)
        chunk_len = min(budget, seq.num_prompt_tokens - start)
        chunk = seq.prompt_token_ids[start : start + chunk_len]

        logits = self.model.continue_prefill(
            self.cache, seq.seq_id, chunk, lora_id=seq.lora_id, lora_registry=self.lora_registry
        )
        budget -= chunk_len

        if self.cache.context_len(seq.seq_id) < seq.num_prompt_tokens:
            return budget, False

        token, logprob = sample(
            logits[:, -1, :], [seq.sampling_params], generated_ids=[seq.output_token_ids], token_masks=[self._token_mask(seq)]
        )
        self.cache.register_prefix_blocks(seq.seq_id, seq.prompt_token_ids)
        seq.status = SeqStatus.RUNNING
        seq.append_token(token.item(), logprob.item())
        return budget, True

    def _prefill(self, seq: Sequence) -> None:
        # Always the prefix-aware path -- it's a strict superset of plain
        # prefill_with_cache, byte-identical when the cache has prefix
        # caching disabled or nothing matches (proven in
        # test_no_match_path_is_identical_to_original_prefill).
        logits = self.model.prefill_with_cache_and_prefix_reuse(
            self.cache, seq.seq_id, seq.prompt_token_ids, lora_id=seq.lora_id, lora_registry=self.lora_registry
        )
        token, logprob = sample(
            logits[:, -1, :], [seq.sampling_params], generated_ids=[seq.output_token_ids], token_masks=[self._token_mask(seq)]
        )
        seq.status = SeqStatus.RUNNING
        seq.append_token(token.item(), logprob.item())

    def _token_mask(self, seq: Sequence):
        return seq.guide.allowed_token_mask(self.device) if seq.guide is not None else None

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        if self.enable_cpu_swap:
            # Swap: preserve progress (both cache content and already-
            # generated tokens) instead of discarding it.
            self.cache.swap_out(seq.seq_id)
            seq.status = SeqStatus.SWAPPED
            self.swapped.insert(0, seq)
        else:
            self.cache.free_sequence(seq.seq_id)
            seq.output_token_ids = []
            seq.status = SeqStatus.WAITING
            self.waiting.insert(0, seq)

    def _try_swap_in(self) -> None:
        """Resumes swapped-out sequences (priority over new admissions, since
        they're already in progress) as capacity allows. A resumed sequence
        becomes RUNNING immediately -- its cache is fully restored, so unlike
        a freshly-completed prefill it needs no new sampled token before
        joining this step's regular decode batch.
        """
        i = 0
        while i < len(self.swapped):
            seq = self.swapped[i]
            needed = self._blocks_needed(self.cache.swapped_context_len[seq.seq_id])
            available = self.cache.allocator.num_free - self._pledged_blocks()
            if needed > available:
                break  # FIFO: don't starve the head-of-line swapped request
            self.swapped.pop(i)
            self.cache.swap_in(seq.seq_id)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)

    def _maybe_propose_draft(self, seq: Sequence) -> list[int] | None:
        """None whenever speculative decoding wouldn't be exact for this
        sequence (see module docstring for why temperature/repetition_penalty/
        guide gate this) or whenever n-gram lookup finds nothing to propose.
        """
        if seq.sampling_params.temperature != 0.0 or seq.sampling_params.repetition_penalty != 1.0:
            return None
        if seq.guide is not None:
            return None
        history = seq.prompt_token_ids + seq.output_token_ids
        return propose_ngram_draft(history, self.speculative_ngram_size, self.speculative_max_draft_len)

    def _speculative_decode_one(self, seq: Sequence, draft: list[int]) -> bool:
        """Verifies `draft` against the target model in one continue_prefill
        call and appends whatever's accepted. Returns False if verification
        itself hit cache exhaustion (seq gets preempted, same as a regular
        decode OOM would) -- True otherwise, regardless of how many of the
        draft tokens were actually accepted (even zero is a normal outcome,
        just means the draft happened to be wrong).
        """
        chunk = [seq.last_token_id] + draft
        try:
            logits = self.model.continue_prefill(
                self.cache, seq.seq_id, chunk, lora_id=seq.lora_id, lora_registry=self.lora_registry
            )
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            self._preempt(seq)
            return False

        row_logits = logits[0].float()  # (len(draft)+1, vocab)
        logprobs_full = row_logits.log_softmax(dim=-1)
        predicted = row_logits.argmax(dim=-1).tolist()

        num_draft = len(draft)
        matched = 0
        while matched < num_draft and predicted[matched] == draft[matched]:
            matched += 1
        final_token = predicted[matched]  # correction if matched < num_draft, else the free bonus token
        accepted_tokens = draft[:matched] + [final_token]

        kept = 0
        for j, tok in enumerate(accepted_tokens):
            seq.append_token(tok, logprobs_full[j, tok].item())
            kept += 1
            if seq.is_finished():
                break

        # Discard whatever was tentatively written for positions beyond what
        # actually got kept (rejected draft tail, or an early EOS cutting a
        # run of otherwise-accepted drafts short).
        rollback = num_draft - min(kept, matched)
        if rollback > 0:
            self.cache.rollback(seq.seq_id, rollback)

        return True

    def _decode_active(self, active: list[Sequence]) -> None:
        """Mutates `active` in place, preempting sequences off the end of it
        if the cache runs out of blocks. If every sequence ends up preempted,
        there's simply nothing left to decode this step.

        Sequences with a lora_id set are split into their own eager-path
        sub-batch even when a graph_decoder is configured (see __init__'s
        docstring on why LoRA can't ride the graph-decoded path); the two
        sub-batches' logits are concatenated back in matching order so the
        rest of this method (and the caller) sees one aligned batch.
        """
        while active:
            try:
                use_graph = self.graph_decoder is not None
                graph_batch = [s for s in active if use_graph and s.lora_id is None]
                eager_batch = [s for s in active if not use_graph or s.lora_id is not None]

                logits_parts = []
                ordered: list[Sequence] = []
                if graph_batch:
                    seq_ids = [s.seq_id for s in graph_batch]
                    logits_parts.append(self.graph_decoder.decode(seq_ids, [s.last_token_id for s in graph_batch]))
                    ordered.extend(graph_batch)
                if eager_batch:
                    seq_ids = [s.seq_id for s in eager_batch]
                    token_ids = torch.tensor([[s.last_token_id] for s in eager_batch], device=self.device)
                    lora_ids = [s.lora_id for s in eager_batch]
                    logits_parts.append(
                        self.model.decode_step_batch(
                            self.cache, seq_ids, token_ids, lora_ids=lora_ids, lora_registry=self.lora_registry
                        )
                    )
                    ordered.extend(eager_batch)

                logits = torch.cat(logits_parts, dim=0) if len(logits_parts) > 1 else logits_parts[0]
                active[:] = ordered
                break
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise
                victim = active.pop()  # simplistic policy: preempt the most recently added
                self._preempt(victim)
        else:
            return

        params = [s.sampling_params for s in active]
        generated_ids = [s.output_token_ids for s in active]
        token_masks = [self._token_mask(s) for s in active]
        tokens, logprobs = sample(logits[:, -1, :], params, generated_ids, token_masks=token_masks)
        for seq, token, logprob in zip(active, tokens.tolist(), logprobs.tolist()):
            seq.append_token(token, logprob)

    def _step_beam_groups(self) -> None:
        """Advances every active beam group by exactly one token each (no
        admission control or preemption for this lane -- a beam group's
        transient scratch-sequence cache usage is freed within the same
        step it's allocated, see BeamGroup.step; if concurrent regular +
        beam-search load genuinely exceeds capacity mid-step, the same
        "out of memory" RuntimeError cache.reserve() always raises simply
        propagates rather than being caught and retried here).
        """
        still_active = []
        for group in self.beam_groups:
            group.step(self.model, self.cache, lora_registry=self.lora_registry)
            if group.is_done():
                self.completed_beam_groups.append(group)
            else:
                still_active.append(group)
        self.beam_groups = still_active

    @torch.inference_mode()
    def step(self) -> list[Sequence]:
        """Runs one scheduler iteration. Returns every sequence that got a
        new token this step -- ones whose prefill just completed (whether in
        one shot or as the final chunk of a chunked prefill) and existing
        running ones that just decoded. A sequence mid-chunked-prefill
        contributes no token yet and isn't included. Check `.is_finished()`
        on the returned sequences to know which ones ended (prefill alone
        can finish a sequence immediately, e.g. max_new_tokens == 1 or an
        instant EOS/stop token).
        """
        self._try_swap_in()
        completed_prefills = self._step_prefill_work()
        completed_ids = {id(s) for s in completed_prefills}

        # Exclude just-completed prefills from this step's decode: they
        # already got one token, so each sequence advances by exactly one
        # token per step -- important for streaming (one queue push per
        # token) and simpler to reason about generally.
        active = [s for s in self.running if s.status == SeqStatus.RUNNING and id(s) not in completed_ids]

        speculative_done = []
        regular_batch = []
        for seq in active:
            draft = self._maybe_propose_draft(seq) if self.enable_speculative_decoding else None
            if draft:
                if self._speculative_decode_one(seq, draft):
                    speculative_done.append(seq)
                # else: OOM'd and got preempted inside _speculative_decode_one;
                # it's already removed from self.running, nothing more to do.
            else:
                regular_batch.append(seq)

        if regular_batch:
            self._decode_active(regular_batch)

        updated = completed_prefills + speculative_done + regular_batch

        finished = [s for s in self.running if s.status == SeqStatus.FINISHED]
        for seq in finished:
            self.cache.free_sequence(seq.seq_id)
        self.running = [s for s in self.running if s.status != SeqStatus.FINISHED]

        self._step_beam_groups()
        return updated

    def run_to_completion(self, max_steps: int = 10_000) -> list[Sequence]:
        finished: list[Sequence] = []
        for _ in range(max_steps):
            if not self.has_unfinished_requests():
                break
            finished.extend(s for s in self.step() if s.is_finished())
        return finished
