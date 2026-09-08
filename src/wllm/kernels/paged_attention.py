import os

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
    """
    return _get_ext().paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, scale)
