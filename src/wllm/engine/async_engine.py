"""Async wrapper around the Scheduler: runs the (synchronous, GPU-bound)
step() loop as a background asyncio task and fans generated tokens out to
per-request queues as text deltas, via the incremental detokenizer. This is
what an HTTP server (P1.7) streams from -- each request gets an async
generator of text chunks instead of having to poll the scheduler itself.

Each scheduler.step() is a blocking call (real GPU compute dominates it, same
as vLLM's own engine loop) -- it isn't split across awaits internally, but the
loop yields to the event loop between steps so other asyncio tasks (request
submission, HTTP handlers) still get scheduled.
"""
from __future__ import annotations

import asyncio
import itertools

from wllm.engine.beam_search import BeamGroup, BeamSearchParams
from wllm.engine.detokenizer import IncrementalDetokenizer
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.sampling import SamplingParams
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence

# Sentinel pushed onto a request's queue to signal the stream is over.
_DONE = object()


class AsyncEngine:
    def __init__(
        self,
        model,
        cache: KVCacheManager,
        tokenizer,
        device: str = "cuda",
        graph_decoder=None,
        max_prefill_tokens_per_step: int | None = None,
        enable_cpu_swap: bool = False,
        enable_speculative_decoding: bool = False,
        speculative_ngram_size: int = 3,
        speculative_max_draft_len: int = 4,
        lora_registry=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lora_registry = lora_registry
        self.scheduler = Scheduler(
            model,
            cache,
            device=device,
            graph_decoder=graph_decoder,
            max_prefill_tokens_per_step=max_prefill_tokens_per_step,
            enable_cpu_swap=enable_cpu_swap,
            enable_speculative_decoding=enable_speculative_decoding,
            speculative_ngram_size=speculative_ngram_size,
            speculative_max_draft_len=speculative_max_draft_len,
            lora_registry=lora_registry,
        )
        self._seq_id_counter = itertools.count()
        self._queues: dict[int, asyncio.Queue] = {}
        self._detokenizers: dict[int, IncrementalDetokenizer] = {}
        self._reported_count: dict[int, int] = {}
        self._beam_group_id_counter = itertools.count()
        self._beam_futures: dict[int, asyncio.Future] = {}
        self._loop_task: asyncio.Task | None = None
        self._idle_poll_interval = 0.005

    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.ensure_future(self._run_loop())

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def _run_loop(self) -> None:
        """If this raises, nobody awaits the task until stop() is called --
        without the try/except below, every pending generate() call would
        hang forever on queue.get() waiting for a _DONE that will never
        arrive. Instead, propagate the failure to every waiting caller.
        """
        try:
            while True:
                if self.scheduler.has_unfinished_requests():
                    updated = self.scheduler.step()
                    for seq in updated:
                        self._on_token(seq)
                    self._drain_completed_beam_groups()
                else:
                    await asyncio.sleep(self._idle_poll_interval)
                await asyncio.sleep(0)  # yield to the event loop even on a busy step
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._fail_all_pending(e)
            raise

    def _fail_all_pending(self, exc: Exception) -> None:
        for queue in self._queues.values():
            queue.put_nowait(exc)
        self._queues.clear()
        self._detokenizers.clear()
        for future in self._beam_futures.values():
            if not future.done():
                future.set_exception(exc)
        self._beam_futures.clear()

    def _drain_completed_beam_groups(self) -> None:
        completed = self.scheduler.completed_beam_groups
        self.scheduler.completed_beam_groups = []
        for group in completed:
            future = self._beam_futures.get(group.group_id)
            if future is not None and not future.done():
                future.set_result(group.result)

    def _on_token(self, seq: Sequence) -> None:
        """Pushes every token new since the last call, not just the latest
        one -- speculative decoding can accept several tokens in a single
        scheduler step (that's the whole point of it), so "just look at
        output_token_ids[-1]" would silently drop the ones in between.
        """
        queue = self._queues.get(seq.seq_id)
        if queue is None:
            return  # caller already stopped listening (e.g. cancelled)
        detok = self._detokenizers[seq.seq_id]
        already_reported = self._reported_count.get(seq.seq_id, 0)
        for token_id in seq.output_token_ids[already_reported:]:
            queue.put_nowait(detok.add_token(token_id))
        self._reported_count[seq.seq_id] = len(seq.output_token_ids)
        if seq.is_finished():
            queue.put_nowait(_DONE)
            del self._queues[seq.seq_id]
            del self._detokenizers[seq.seq_id]
            self._reported_count.pop(seq.seq_id, None)

    async def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        sampling_params: SamplingParams | None = None,
        eos_token_id: int | None = None,
        stop_token_ids: frozenset[int] = frozenset(),
        guide=None,
        lora_id: str | None = None,
    ):
        """Async generator yielding text deltas as they're produced.
        `guide`, if given, is a JSONSchemaGuide constraining every token this
        sequence generates to a JSON schema (see guided_decoding.py).
        `lora_id`, if given, must already be loaded in this engine's
        lora_registry -- it selects which adapter's deltas apply to every
        linear layer this sequence's forward passes touch (see lora.py).
        """
        seq_id = next(self._seq_id_counter)
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[seq_id] = queue
        self._detokenizers[seq_id] = IncrementalDetokenizer(self.tokenizer)

        seq = Sequence(
            seq_id=seq_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
            sampling_params=sampling_params or SamplingParams(),
            guide=guide,
            lora_id=lora_id,
        )
        self.scheduler.add_request(seq)
        completed_normally = False

        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    completed_normally = True
                    break
                if isinstance(item, BaseException):
                    completed_normally = True  # engine already tore everything down
                    raise item
                yield item
        finally:
            # If the consumer stops early (stop-string match, client
            # disconnect) the sequence is still running in the scheduler and
            # holding cache blocks -- cancel it rather than leaking them.
            if not completed_normally:
                self.scheduler.cancel(seq_id)
            self._queues.pop(seq_id, None)
            self._detokenizers.pop(seq_id, None)
            self._reported_count.pop(seq_id, None)

    async def generate_beam_search(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float = 1.0,
        eos_token_id: int | None = None,
        stop_token_ids: frozenset[int] = frozenset(),
        lora_id: str | None = None,
    ) -> tuple[str, str]:
        """Unlike generate(), this is a plain coroutine, not a streaming
        async generator: beam search's winning sequence isn't known until
        the whole group finishes (a leading candidate can still be
        discarded later as a better-scoring one emerges), so there's
        nothing meaningful to stream token-by-token. Returns
        (winning beam's decoded text, finish_reason).
        """
        group_id = next(self._beam_group_id_counter)
        group = BeamGroup(
            group_id=group_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            params=BeamSearchParams(num_beams=num_beams, length_penalty=length_penalty),
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
            lora_id=lora_id,
        )
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._beam_futures[group_id] = future
        self.scheduler.add_beam_request(group)
        try:
            token_ids = await future
        except asyncio.CancelledError:
            if group in self.scheduler.beam_groups:
                self.scheduler.beam_groups.remove(group)
            raise
        finally:
            self._beam_futures.pop(group_id, None)
        return self.tokenizer.decode(token_ids, skip_special_tokens=True), group.finish_reason
