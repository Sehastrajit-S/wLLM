import os

import torch
from torch.utils.cpp_extension import load

from wllm.kernels._msvc_env import ensure_msvc_on_path

_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        ensure_msvc_on_path()
        _ext = load(
            name="wllm_paged_attention",
            sources=[os.path.join(_CSRC_DIR, "paged_attention.cu")],
            extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        )
    return _ext


def paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, scale):
    """q: (num_seqs, num_heads, head_dim) float32
    k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim) float32
    block_tables: (num_seqs, max_blocks_per_seq) int32
    context_lens: (num_seqs,) int32
    returns: (num_seqs, num_heads, head_dim) float32

    Dispatches on q's device: the custom CUDA kernel when it's a CUDA
    tensor (the fast path everything is tuned around), a plain-PyTorch
    fallback otherwise (see _paged_attention_decode_cpu below) so decode --
    and so the whole engine -- also runs correctly, just without this
    kernel's speed or CUDA graph capture, on a CPU-only machine. Neither
    MSVC nor the CUDA Toolkit is touched at all on that path: _get_ext()
    (the JIT compile) is only ever called from the branch below.
    """
    if q.is_cuda:
        return _get_ext().paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, scale)
    return _paged_attention_decode_cpu(q, k_cache, v_cache, block_tables, context_lens, scale)


def _paged_attention_decode_cpu(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Reference-quality (not performance-tuned) implementation of the exact
    same math as the CUDA kernel: gather each sequence's valid cached K/V
    tokens by walking its block table, then a single-query attention over
    that context. A per-sequence Python loop, not batched -- correctness for
    CPU inference is the goal here, not competing with the CUDA kernel.
    """
    num_seqs, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    n_rep = num_heads // num_kv_heads

    out = torch.empty(num_seqs, num_heads, head_dim, dtype=q.dtype, device=q.device)
    for i in range(num_seqs):
        ctx_len = int(context_lens[i].item())
        n_blocks = (ctx_len + block_size - 1) // block_size
        block_ids = block_tables[i, :n_blocks].long()

        k = k_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:ctx_len]  # (ctx_len, num_kv_heads, head_dim)
        v = v_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:ctx_len]
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)  # (ctx_len, num_heads, head_dim)
            v = v.repeat_interleave(n_rep, dim=1)

        scores = torch.einsum("hd,chd->hc", q[i], k) * scale  # (num_heads, ctx_len)
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out[i] = torch.einsum("hc,chd->hd", weights, v)

    return out
