import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from wllm.kernels.paged_attention import paged_attention_decode

BLOCK_SIZE = 16
HEAD_DIM = 64
NUM_HEADS = 14
NUM_KV_HEADS = 2
NUM_QUERIES_PER_KV = NUM_HEADS // NUM_KV_HEADS


def reference_paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Pure-PyTorch reference: gather each sequence's cached K/V through its
    block table and run standard (non-streaming) softmax attention. This is
    what the CUDA kernel's streaming online-softmax result gets diffed against.
    """
    num_seqs, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    n_rep = num_heads // num_kv_heads
    out = torch.empty_like(q)

    for s in range(num_seqs):
        ctx_len = context_lens[s].item()
        block_ids = block_tables[s, : (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE]
        k = k_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:ctx_len]  # (ctx_len, kv_heads, head_dim)
        v = v_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:ctx_len]

        k = k.repeat_interleave(n_rep, dim=1)  # (ctx_len, num_heads, head_dim)
        v = v.repeat_interleave(n_rep, dim=1)

        qs = q[s]  # (num_heads, head_dim)
        scores = torch.einsum("hd,thd->ht", qs, k) * scale  # (num_heads, ctx_len)
        probs = torch.softmax(scores, dim=-1)
        out[s] = torch.einsum("ht,thd->hd", probs, v)

    return out


def make_random_case(context_lens: list[int], seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda"
    num_seqs = len(context_lens)
    max_ctx = max(context_lens)
    max_blocks_per_seq = (max_ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * max_blocks_per_seq + 4  # some spare

    q = torch.randn(num_seqs, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    v_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float32)

    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, device=device, dtype=torch.int32)
    next_block = 0
    for s in range(num_seqs):
        n_blocks = (context_lens[s] + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(n_blocks):
            block_tables[s, b] = next_block
            next_block += 1

    context_lens_t = torch.tensor(context_lens, device=device, dtype=torch.int32)
    scale = 1.0 / (HEAD_DIM**0.5)
    return q, k_cache, v_cache, block_tables, context_lens_t, scale


@pytest.mark.parametrize(
    "context_lens",
    [
        [1],
        [5],
        [16],
        [17],
        [33],
        [100],
        [1, 5, 16, 17, 33, 100],
    ],
)
def test_paged_attention_matches_reference(context_lens):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    q, k_cache, v_cache, block_tables, context_lens_t, scale = make_random_case(context_lens)

    kernel_out = paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens_t, scale)
    ref_out = reference_paged_attention(q, k_cache, v_cache, block_tables, context_lens_t, scale)

    diff = (kernel_out - ref_out).abs()
    assert diff.max().item() < 1e-3, f"max diff {diff.max().item()} for context_lens={context_lens}"
