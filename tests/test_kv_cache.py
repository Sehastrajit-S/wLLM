import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from wllm.engine.kv_cache import BlockAllocator, KVCacheManager


def test_block_allocator_reuses_freed_blocks():
    alloc = BlockAllocator(num_blocks=4)
    a, b = alloc.allocate(), alloc.allocate()
    assert alloc.num_free == 2

    alloc.free(a)
    assert alloc.num_free == 3

    c = alloc.allocate()
    assert c == a  # LIFO free-list reuse
    assert alloc.num_free == 2
    assert len({a, b, c}) == 2  # a and c are the same block


def test_block_allocator_raises_when_exhausted():
    alloc = BlockAllocator(num_blocks=1)
    alloc.allocate()
    with pytest.raises(RuntimeError):
        alloc.allocate()


def test_reserve_grows_block_table_at_boundaries():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cache = KVCacheManager(
        num_layers=1, num_blocks=8, block_size=4, num_kv_heads=1, head_dim=8, device="cuda", dtype=torch.float32
    )
    cache.create_sequence(seq_id=0)

    positions = cache.reserve(0, 4)  # exactly fills block 0
    assert len(cache.sequences[0].block_table) == 1
    assert [p[0] for p in positions] == [cache.sequences[0].block_table[0]] * 4
    assert [p[1] for p in positions] == [0, 1, 2, 3]
    cache.advance(0, 4)

    positions = cache.reserve(0, 1)  # must allocate a second block
    assert len(cache.sequences[0].block_table) == 2
    assert positions[0] == (cache.sequences[0].block_table[1], 0)
    cache.advance(0, 1)

    assert cache.context_len(0) == 5


def test_free_sequence_returns_blocks_to_allocator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cache = KVCacheManager(
        num_layers=1, num_blocks=2, block_size=4, num_kv_heads=1, head_dim=8, device="cuda", dtype=torch.float32
    )
    cache.create_sequence(seq_id=0)
    cache.reserve(0, 4)
    cache.advance(0, 4)
    assert cache.allocator.num_free == 1

    cache.free_sequence(0)
    assert cache.allocator.num_free == 2


def test_match_prefix_disabled_by_default_returns_no_match():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cache = KVCacheManager(
        num_layers=1, num_blocks=8, block_size=4, num_kv_heads=1, head_dim=8, device="cuda", dtype=torch.float32
    )
    blocks, n = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
    assert blocks == []
    assert n == 0


def test_register_then_match_prefix_roundtrip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cache = KVCacheManager(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=1,
        head_dim=8,
        device="cuda",
        dtype=torch.float32,
        enable_prefix_caching=True,
    )
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # exactly 3 blocks
    cache.create_sequence(0)
    cache.reserve(0, len(tokens))
    cache.advance(0, len(tokens))
    cache.register_prefix_blocks(0, tokens)

    # A fresh request with the identical token prefix should match all but
    # the trailing partial/edge block per match_prefix's "leave one token
    # unmatched" rule -- here len(tokens) is block-aligned (12 = 3*4), so
    # the last of the 3 blocks is deliberately left unmatched too.
    matched_blocks, num_matched = cache.match_prefix(tokens)
    assert num_matched == 8  # first 2 of 3 blocks
    assert matched_blocks == cache.sequences[0].block_table[:2]


def test_match_prefix_diverges_at_first_different_block():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cache = KVCacheManager(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=1,
        head_dim=8,
        device="cuda",
        dtype=torch.float32,
        enable_prefix_caching=True,
    )
    tokens = list(range(16))  # 4 blocks
    cache.create_sequence(0)
    cache.reserve(0, len(tokens))
    cache.advance(0, len(tokens))
    cache.register_prefix_blocks(0, tokens)

    different = tokens[:4] + [999, 999, 999, 999] + tokens[8:]  # block 1 diverges
    matched_blocks, num_matched = cache.match_prefix(different)
    assert num_matched == 4  # only the first (matching) block
    assert matched_blocks == cache.sequences[0].block_table[:1]


def test_create_sequence_from_prefix_shares_refcounted_blocks():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cache = KVCacheManager(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=8,
        device="cuda",
        dtype=torch.float32,
        enable_prefix_caching=True,
    )
    tokens = list(range(8))  # 2 blocks
    cache.create_sequence(0)
    cache.reserve(0, len(tokens))
    cache.advance(0, len(tokens))
    cache.register_prefix_blocks(0, tokens)
    shared_blocks = cache.sequences[0].block_table[:1]  # first block only (2nd left unmatched)

    assert cache.allocator.num_free == 2  # 4 total - 2 used by seq 0

    cache.create_sequence_from_prefix(1, shared_blocks)
    assert cache.context_len(1) == 4
    assert cache.allocator.num_free == 2  # no new allocation -- shared, not copied
    assert cache.allocator.ref_counts[shared_blocks[0]] == 2

    # freeing seq 0 must NOT free the still-shared block
    cache.free_sequence(0)
    assert cache.allocator.ref_counts[shared_blocks[0]] == 1
    assert shared_blocks[0] not in cache.allocator.free_blocks

    # freeing seq 1 (the last reference) does free it
    cache.free_sequence(1)
    assert cache.allocator.ref_counts[shared_blocks[0]] == 0
    assert shared_blocks[0] in cache.allocator.free_blocks


def test_freeing_last_reference_removes_stale_hash_entry():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cache = KVCacheManager(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=8,
        device="cuda",
        dtype=torch.float32,
        enable_prefix_caching=True,
    )
    tokens = list(range(8))
    cache.create_sequence(0)
    cache.reserve(0, len(tokens))
    cache.advance(0, len(tokens))
    cache.register_prefix_blocks(0, tokens)
    assert len(cache.hash_to_block) == 2  # both complete blocks registered (the unmatched-tail rule is match_prefix-only)

    cache.free_sequence(0)
    assert cache.hash_to_block == {}
    assert cache.block_hash_info == {}

    # a later sequence with the same tokens must recompute from scratch, not
    # hit a stale mapping to a block that's since been reused for something else
    matched_blocks, num_matched = cache.match_prefix(tokens)
    assert matched_blocks == []
    assert num_matched == 0


def test_rollback_shrinks_context_len_and_position_is_correctly_rewritable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cache = KVCacheManager(
        num_layers=1, num_blocks=4, block_size=4, num_kv_heads=1, head_dim=8, device="cuda", dtype=torch.float32
    )
    cache.create_sequence(0)
    positions = cache.reserve(0, 4)
    cache.write(0, positions, torch.ones(4, 1, 8, device="cuda"), torch.ones(4, 1, 8, device="cuda"))
    cache.advance(0, 4)
    assert cache.context_len(0) == 4

    cache.rollback(0, 2)
    assert cache.context_len(0) == 2

    # the rolled-back positions must be cleanly rewritable with different
    # content -- reserve() should hand back the same (block_id, slot) pairs
    # as before, not skip past them.
    new_positions = cache.reserve(0, 2)
    assert new_positions == positions[2:4]
    cache.write(0, new_positions, torch.zeros(2, 1, 8, device="cuda"), torch.zeros(2, 1, 8, device="cuda"))
    cache.advance(0, 2)
    assert cache.context_len(0) == 4

    for block_id, slot in positions[2:4]:
        assert torch.equal(cache.k_caches[0][block_id, slot], torch.zeros(1, 8, device="cuda"))


def test_write_then_read_roundtrip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cache = KVCacheManager(
        num_layers=1, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=8, device="cuda", dtype=torch.float32
    )
    cache.create_sequence(seq_id=0)
    positions = cache.reserve(0, 3)

    key = torch.randn(3, 2, 8, device="cuda")
    value = torch.randn(3, 2, 8, device="cuda")
    cache.write(layer_idx=0, positions=positions, key=key, value=value)
    cache.advance(0, 3)

    for i, (block_id, slot) in enumerate(positions):
        assert torch.equal(cache.k_caches[0][block_id, slot], key[i])
        assert torch.equal(cache.v_caches[0][block_id, slot], value[i])


def test_swap_out_then_swap_in_restores_identical_kv_and_context_len():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cache = KVCacheManager(
        num_layers=2, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=8, device="cuda", dtype=torch.float32
    )
    cache.create_sequence(seq_id=0)
    positions = cache.reserve(0, 6)  # spans 2 blocks
    for layer in range(2):
        key = torch.randn(6, 2, 8, device="cuda")
        value = torch.randn(6, 2, 8, device="cuda")
        cache.write(layer, positions, key, value)
    cache.advance(0, 6)

    original_blocks = list(cache.sequences[0].block_table)
    k_before = [cache.k_caches[i][original_blocks].clone() for i in range(2)]
    v_before = [cache.v_caches[i][original_blocks].clone() for i in range(2)]

    free_before_swap = cache.allocator.num_free
    cache.swap_out(0)
    assert 0 not in cache.sequences
    assert cache.is_swapped(0)
    assert cache.allocator.num_free == free_before_swap + len(original_blocks)

    cache.swap_in(0)
    assert not cache.is_swapped(0)
    assert cache.context_len(0) == 6

    new_blocks = cache.sequences[0].block_table
    assert len(new_blocks) == len(original_blocks)
    for i in range(2):
        assert torch.equal(cache.k_caches[i][new_blocks], k_before[i])
        assert torch.equal(cache.v_caches[i][new_blocks], v_before[i])


def test_discard_swapped_frees_host_memory_without_restoring():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cache = KVCacheManager(
        num_layers=1, num_blocks=4, block_size=4, num_kv_heads=1, head_dim=8, device="cuda", dtype=torch.float32
    )
    cache.create_sequence(seq_id=0)
    positions = cache.reserve(0, 4)
    cache.write(0, positions, torch.randn(4, 1, 8, device="cuda"), torch.randn(4, 1, 8, device="cuda"))
    cache.advance(0, 4)

    cache.swap_out(0)
    assert cache.is_swapped(0)
    cache.discard_swapped(0)
    assert not cache.is_swapped(0)
    assert 0 not in cache.swapped_context_len
