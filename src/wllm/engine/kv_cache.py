"""Block-based KV cache: the allocator + per-sequence block tables that make
PagedAttention possible. One block table is shared across all layers of a
sequence (a block index means "these token slots," identically for every
layer); each layer owns its own K/V storage tensors.

Automatic prefix caching (optional, via `enable_prefix_caching`): blocks are
refcounted rather than owned by one sequence, and a block whose exact token
content matches an earlier completed block (via a hash chained from the
start of the sequence, so the match is "this exact prefix," not just "this
exact block's tokens in isolation") gets reused directly -- no recomputation,
no rewrite. Reuse only ever happens at whole-block granularity: a sequence's
own new tokens always land in a freshly allocated block, never inside a
block it doesn't exclusively own, so there's no copy-on-write to reason
about. A freed block's hash entry is removed the moment its refcount drops
to zero, so a later sequence can never hit a stale mapping to a block that's
since been overwritten with different content.
"""
from __future__ import annotations

import torch


def _hash_block(parent_hash: int | None, token_chunk: tuple[int, ...]) -> int:
    return hash((parent_hash, token_chunk))


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks: list[int] = list(range(num_blocks))
        self.ref_counts: list[int] = [0] * num_blocks

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("KV cache out of memory: no free blocks left")
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def add_ref(self, block_id: int) -> None:
        self.ref_counts[block_id] += 1

    def free(self, block_id: int) -> bool:
        """Returns True if this was the last reference (block is now free)."""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] <= 0:
            self.ref_counts[block_id] = 0
            self.free_blocks.append(block_id)
            return True
        return False

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


class SequenceState:
    def __init__(self):
        self.block_table: list[int] = []
        self.context_len: int = 0


class KVCacheManager:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        enable_prefix_caching: bool = False,
    ):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.allocator = BlockAllocator(num_blocks)
        self.k_caches = [
            torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.sequences: dict[int, SequenceState] = {}

        # Prefix-cache bookkeeping (unused unless enable_prefix_caching).
        self.hash_to_block: dict[int, int] = {}
        self.block_hash_info: dict[int, tuple[int, tuple[int, ...]]] = {}  # block_id -> (hash, token_chunk)

        # CPU-swap bookkeeping (unused unless swap_out/swap_in are called).
        self.swapped_kv: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}  # seq_id -> per-layer (k_cpu, v_cpu)
        self.swapped_context_len: dict[int, int] = {}

    def create_sequence(self, seq_id: int) -> None:
        self.sequences[seq_id] = SequenceState()

    def create_sequence_from_prefix(self, seq_id: int, prefix_block_ids: list[int]) -> None:
        """Like create_sequence, but pre-populates the block table with
        already-cached, reused blocks (refcounted, not copied).
        """
        state = SequenceState()
        for block_id in prefix_block_ids:
            self.allocator.add_ref(block_id)
            state.block_table.append(block_id)
        state.context_len = len(prefix_block_ids) * self.block_size
        self.sequences[seq_id] = state

    def _free_block(self, block_id: int) -> None:
        if self.allocator.free(block_id) and block_id in self.block_hash_info:
            block_hash, _ = self.block_hash_info.pop(block_id)
            if self.hash_to_block.get(block_hash) == block_id:
                del self.hash_to_block[block_hash]

    def free_sequence(self, seq_id: int) -> None:
        state = self.sequences.pop(seq_id)
        for block_id in state.block_table:
            self._free_block(block_id)

    def swap_out(self, seq_id: int) -> None:
        """Preemption that preserves progress instead of discarding it: copies
        this sequence's entire KV cache to host RAM (one transfer per layer,
        synchronous -- correctness-first, an async/pinned-memory version is a
        real future optimization but not implemented here) and frees its GPU
        blocks. swap_in() later restores it byte-for-byte with no recompute.
        """
        state = self.sequences.pop(seq_id)
        block_ids = state.block_table
        idx = torch.tensor(block_ids, dtype=torch.long, device=self.k_caches[0].device)

        per_layer = []
        for k_cache, v_cache in zip(self.k_caches, self.v_caches):
            per_layer.append((k_cache[idx].cpu(), v_cache[idx].cpu()))
        self.swapped_kv[seq_id] = per_layer
        self.swapped_context_len[seq_id] = state.context_len

        for block_id in block_ids:
            self._free_block(block_id)

    def swap_in(self, seq_id: int) -> None:
        """Restores a sequence swapped out by swap_out(): allocates fresh GPU
        blocks and copies the host-side snapshot back in, then re-registers
        the sequence with its original context_len (its generated tokens
        live on the Sequence object, untouched by any of this -- only the
        KV cache itself needed saving/restoring).
        """
        context_len = self.swapped_context_len.pop(seq_id)
        per_layer = self.swapped_kv.pop(seq_id)
        num_blocks_needed = (context_len + self.block_size - 1) // self.block_size

        block_table = [self.allocator.allocate() for _ in range(num_blocks_needed)]
        idx = torch.tensor(block_table, dtype=torch.long, device=self.k_caches[0].device)
        for (k_cache, v_cache), (k_cpu, v_cpu) in zip(zip(self.k_caches, self.v_caches), per_layer):
            k_cache[idx] = k_cpu.to(k_cache.device)
            v_cache[idx] = v_cpu.to(v_cache.device)

        state = SequenceState()
        state.block_table = block_table
        state.context_len = context_len
        self.sequences[seq_id] = state

    def discard_swapped(self, seq_id: int) -> None:
        """Drops a swapped-out sequence's host-side snapshot without
        restoring it (e.g. the request was cancelled while swapped out).
        """
        self.swapped_kv.pop(seq_id, None)
        self.swapped_context_len.pop(seq_id, None)

    def is_swapped(self, seq_id: int) -> bool:
        return seq_id in self.swapped_kv

    def context_len(self, seq_id: int) -> int:
        return self.sequences[seq_id].context_len

    def rollback(self, seq_id: int, num_tokens: int) -> None:
        """Discards the last `num_tokens` positions from a sequence's cached
        context -- used by speculative decoding to drop rejected draft
        tokens' tentatively-written K/V. Nothing needs to be erased: a block
        table position beyond context_len is simply "not written yet" as far
        as reserve()/advance() are concerned, and will be correctly
        overwritten the next time this sequence actually reaches it. Blocks
        are not deallocated even if a rollback leaves one entirely unused --
        they stay owned by this sequence and get reused by its next real
        token; a minor, deliberately-accepted inefficiency rather than added
        complexity for a rare case.
        """
        self.sequences[seq_id].context_len -= num_tokens

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Walks `token_ids` in whole-block chunks from the start, following
        the hash chain as far as it matches an already-cached block. Returns
        (block_ids_to_reuse, num_matched_tokens). Always leaves at least one
        token unmatched (even on a 100%-identical resubmission) so there's
        always at least one real position to run the forward pass on and get
        logits for the first generated token from.
        """
        if not self.enable_prefix_caching:
            return [], 0

        matched_blocks: list[int] = []
        parent_hash: int | None = None
        num_full_blocks = len(token_ids) // self.block_size
        if len(token_ids) % self.block_size == 0:
            num_full_blocks -= 1  # leave the last block unmatched, see docstring

        for i in range(num_full_blocks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            block_hash = _hash_block(parent_hash, chunk)
            block_id = self.hash_to_block.get(block_hash)
            if block_id is None:
                break
            stored_hash, stored_chunk = self.block_hash_info[block_id]
            if stored_chunk != chunk:
                break  # hash collision -- treat as a miss rather than risk silent corruption
            matched_blocks.append(block_id)
            parent_hash = block_hash

        return matched_blocks, len(matched_blocks) * self.block_size

    def register_prefix_blocks(self, seq_id: int, token_ids: list[int]) -> None:
        """Call once after a sequence's prompt has been prefilled, so any
        newly-completed whole blocks become reusable by future sequences
        sharing the same prefix. Only covers the prompt (not blocks completed
        purely from generated continuation tokens during decode) -- shared
        prompts/system-prompts are where prefix caching earns its keep;
        decode-time registration would add bookkeeping cost with much less
        real-world payoff and is left out of this pass.
        """
        if not self.enable_prefix_caching:
            return

        state = self.sequences[seq_id]
        num_full_blocks = len(token_ids) // self.block_size
        parent_hash: int | None = None
        for i in range(num_full_blocks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            block_hash = _hash_block(parent_hash, chunk)
            block_id = state.block_table[i]
            if block_id not in self.block_hash_info:
                self.hash_to_block[block_hash] = block_id
                self.block_hash_info[block_id] = (block_hash, chunk)
            parent_hash = block_hash

    def gather_prefix_kv(self, layer_idx: int, prefix_block_ids: list[int], prefix_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (k, v) each (prefix_len, num_kv_heads, head_dim), gathered
        from the given layer's cache through the reused blocks -- a
        contiguous view/copy suitable for concatenating with newly-computed
        suffix K/V ahead of a standard (non-paged) attention call.
        """
        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]
        num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
        idx = torch.tensor(prefix_block_ids, dtype=torch.long, device=k_cache.device)
        k = k_cache[idx].reshape(-1, num_kv_heads, head_dim)[:prefix_len]
        v = v_cache[idx].reshape(-1, num_kv_heads, head_dim)[:prefix_len]
        return k, v

    def reserve(self, seq_id: int, num_new_tokens: int) -> list[tuple[int, int]]:
        """Ensures the block table has enough blocks for `num_new_tokens` more
        tokens (starting right after the current context_len) and returns the
        (block_id, slot) each new token should land in. Does NOT advance
        context_len -- call advance() once all layers have written.
        """
        state = self.sequences[seq_id]
        positions = []
        pos = state.context_len
        for _ in range(num_new_tokens):
            block_idx_in_seq = pos // self.block_size
            slot = pos % self.block_size
            if block_idx_in_seq == len(state.block_table):
                state.block_table.append(self.allocator.allocate())
            positions.append((state.block_table[block_idx_in_seq], slot))
            pos += 1
        return positions

    def write(
        self,
        layer_idx: int,
        positions: list[tuple[int, int]],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """key/value: (num_new_tokens, num_kv_heads, head_dim)."""
        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]
        for i, (block_id, slot) in enumerate(positions):
            k_cache[block_id, slot] = key[i]
            v_cache[block_id, slot] = value[i]

    def advance(self, seq_id: int, num_new_tokens: int) -> None:
        self.sequences[seq_id].context_len += num_new_tokens

    def block_tables_tensor(self, seq_ids: list[int], device: str = "cuda") -> torch.Tensor:
        max_blocks = max(len(self.sequences[s].block_table) for s in seq_ids)
        out = torch.zeros(len(seq_ids), max_blocks, dtype=torch.int32, device=device)
        for i, s in enumerate(seq_ids):
            bt = self.sequences[s].block_table
            out[i, : len(bt)] = torch.tensor(bt, dtype=torch.int32, device=device)
        return out

    def context_lens_tensor(self, seq_ids: list[int], device: str = "cuda") -> torch.Tensor:
        return torch.tensor([self.sequences[s].context_len for s in seq_ids], dtype=torch.int32, device=device)
