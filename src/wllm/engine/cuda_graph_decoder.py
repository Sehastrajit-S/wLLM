"""Bucketed CUDA graph capture for the decode step, matching vLLM's real
approach: capture one graph per bucket batch size (1, 2, 4, 8, 16, 32, ...),
pick the smallest bucket that fits the active batch, pad up to it, replay.

Profiling this engine's decode step (see scripts/benchmark_decode.py and the
Phase 3 discussion) showed the custom PagedAttention kernel itself costs
~1ms/step out of ~30ms+ wall time -- the real cost is ~880 tiny CUDA kernel
launches per step, each paying ~8us of CPU dispatch overhead that the GPU
then sits idle waiting on. A CUDA graph replays that whole captured sequence
of kernel launches with a single call, eliminating essentially all of that
per-step Python/dispatch overhead.

Padding uses permanent "scratch" sequences (one dummy token, never advanced)
rather than reusing a real sequence's slot for padding rows -- reusing a
real sequence's state for padding would either corrupt it (if written) or
require careful indexing gymnastics to avoid it; a handful of dedicated,
never-touched-after-setup scratch sequences sidesteps all of that.

Falls back to the ordinary eager decode_step_batch (always correct, just
slower) whenever the batch exceeds the largest captured bucket or a
sequence's context has grown past what the graph's fixed-width block_table
buffer can represent -- this is a strict optimization layer, never a
correctness risk.
"""
from __future__ import annotations

import torch

from wllm.engine.kv_cache import KVCacheManager

_SCRATCH_SEQ_ID_BASE = 2**31  # far outside any real seq_id range


class CUDAGraphDecoder:
    def __init__(
        self,
        model,
        cache: KVCacheManager,
        bucket_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        max_blocks_per_seq: int = 256,
        device: str = "cuda",
    ):
        # CUDA graphs are, definitionally, a CUDA-only capability -- there's
        # no portable equivalent to capture on CPU. Failing here with a
        # clear message beats the alternative: without this check, capture()
        # would instead die deep inside torch.cuda.Stream() with a cryptic
        # error unrelated to what the caller actually did wrong. CPU
        # inference just means never constructing a CUDAGraphDecoder at all
        # (Scheduler's graph_decoder is optional, defaulting to None).
        if not torch.device(device).type == "cuda":
            raise ValueError(
                f"CUDAGraphDecoder requires a CUDA device, got {device!r}. "
                "For CPU inference, construct the Scheduler without a graph_decoder instead."
            )
        self.model = model
        self.cache = cache
        self.bucket_sizes = tuple(sorted(bucket_sizes))
        self.max_bucket = self.bucket_sizes[-1]
        self.max_blocks_per_seq = max_blocks_per_seq
        self.device = device
        self.graphs: dict[int, dict] = {}
        self.scratch_seq_ids: list[int] = []
        self._captured = False

    @torch.inference_mode()
    def capture(self) -> None:
        self._setup_scratch_sequences()
        for bucket in self.bucket_sizes:
            self._capture_one(bucket)
        self._captured = True

    def _setup_scratch_sequences(self) -> None:
        n_scratch = self.max_bucket - 1
        dummy_token = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        for i in range(n_scratch):
            seq_id = _SCRATCH_SEQ_ID_BASE + i
            self.model.prefill_with_cache(self.cache, seq_id, dummy_token)
            self.scratch_seq_ids.append(seq_id)

    def _scratch_write_idx(self, seq_id: int) -> int:
        """Scratch sequences are never advanced past context_len 1, so their
        one allocated block's slot 0 is theirs alone to keep overwriting --
        safe as a placeholder write target during warmup/capture and for
        real padding rows, since nothing ever reads it meaningfully.
        """
        block_id = self.cache.sequences[seq_id].block_table[0]
        return block_id * self.cache.block_size

    def _capture_one(self, batch_size: int) -> None:
        device = self.device
        token_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        block_table = torch.zeros(batch_size, self.max_blocks_per_seq, dtype=torch.int32, device=device)
        context_len = torch.ones(batch_size, dtype=torch.int32, device=device)
        write_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        scratch_id = self.scratch_seq_ids[0]
        bt = self.cache.block_tables_tensor([scratch_id] * batch_size, device=device)
        block_table[:, : bt.shape[1]].copy_(bt)
        context_len.fill_(self.cache.context_len(scratch_id))
        write_idx.fill_(self._scratch_write_idx(scratch_id))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model.decode_step_graphable(self.cache, token_ids, position_ids, block_table, context_len, write_idx)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = self.model.decode_step_graphable(
                self.cache, token_ids, position_ids, block_table, context_len, write_idx
            )

        self.graphs[batch_size] = {
            "graph": graph,
            "token_ids": token_ids,
            "position_ids": position_ids,
            "block_table": block_table,
            "context_len": context_len,
            "write_idx": write_idx,
            "logits": static_logits,
        }

    def _fits_captured_graphs(self, seq_ids: list[int]) -> bool:
        if len(seq_ids) > self.max_bucket:
            return False
        for s in seq_ids:
            needed_blocks = len(self.cache.sequences[s].block_table) + 1  # +1: this step may allocate one more
            if needed_blocks > self.max_blocks_per_seq:
                return False
        return True

    @torch.inference_mode()
    def decode(self, seq_ids: list[int], token_ids_list: list[int]) -> torch.Tensor:
        if not self._captured:
            raise RuntimeError("call capture() before decode()")
        if not self._fits_captured_graphs(seq_ids):
            token_ids_tensor = torch.tensor([[t] for t in token_ids_list], device=self.device)
            return self.model.decode_step_batch(self.cache, seq_ids, token_ids_tensor)

        n = len(seq_ids)
        bucket = next(b for b in self.bucket_sizes if b >= n)
        buf = self.graphs[bucket]
        pad = bucket - n
        pad_seq_ids = self.scratch_seq_ids[:pad]

        old_lens = [self.cache.context_len(s) for s in seq_ids]
        real_positions = [self.cache.reserve(s, 1)[0] for s in seq_ids]

        all_seq_ids = seq_ids + pad_seq_ids
        all_token_ids = token_ids_list + [0] * pad
        pad_context_lens = [self.cache.context_len(s) for s in pad_seq_ids]
        all_position_ids = old_lens + pad_context_lens
        all_context_lens = [length + 1 for length in old_lens] + pad_context_lens

        block_table_real = self.cache.block_tables_tensor(all_seq_ids, device=self.device)
        w = block_table_real.shape[1]
        buf["block_table"][:bucket, :w].copy_(block_table_real)
        if w < self.max_blocks_per_seq:
            buf["block_table"][:bucket, w:].zero_()

        buf["context_len"][:bucket].copy_(torch.tensor(all_context_lens, dtype=torch.int32, device=self.device))
        buf["position_ids"][:bucket, 0].copy_(torch.tensor(all_position_ids, dtype=torch.long, device=self.device))
        buf["token_ids"][:bucket, 0].copy_(torch.tensor(all_token_ids, dtype=torch.long, device=self.device))

        write_idx_values = [block_id * self.cache.block_size + slot for block_id, slot in real_positions]
        write_idx_values += [self._scratch_write_idx(s) for s in pad_seq_ids]
        buf["write_idx"][:bucket].copy_(torch.tensor(write_idx_values, dtype=torch.long, device=self.device))

        buf["graph"].replay()

        for s in seq_ids:
            self.cache.advance(s, 1)

        return buf["logits"][:n].clone()
