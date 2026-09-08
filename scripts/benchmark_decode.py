"""Benchmarks decode-step throughput/latency through the real PagedAttention
+ continuous batching path, at several batch sizes. Run before AND after
kernel optimization work to quantify actual speedup -- numbers, not vibes.

Note: there's no way to install real vLLM in this environment for a direct
side-by-side (no official Windows support, which is the whole premise of
this project) -- this measures our own before/after, not a vLLM comparison.

Run: python scripts/benchmark_decode.py [model_id] [comma,separated,batch,sizes]
"""
import sys
import time

sys.path.insert(0, "src")

import torch
from transformers import AutoTokenizer

from wllm.baseline.model import DEFAULT_MODEL_ID
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.qwen2 import load_native

PROMPT = [{"role": "user", "content": "Tell me a short story about a robot who learns to paint."}]
DECODE_STEPS = 64
WARMUP_STEPS = 8


def make_cache(model, num_blocks, block_size=16):
    cfg = model.cfg
    return KVCacheManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device="cuda",
        dtype=torch.float32,
    )


def benchmark_batch_size(model, tokenizer, batch_size: int) -> float:
    """Returns aggregate decode throughput in tokens/sec across `batch_size`
    concurrently-running sequences.
    """
    prompt_ids = tokenizer.apply_chat_template(
        PROMPT, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].tolist()

    cache = make_cache(model, num_blocks=batch_size * 32)
    scheduler = Scheduler(model, cache)
    for i in range(batch_size):
        scheduler.add_request(
            Sequence(seq_id=i, prompt_token_ids=prompt_ids, max_new_tokens=WARMUP_STEPS + DECODE_STEPS + 1)
        )

    # Admit + prefill all of them, then warm up (first steps pay one-time
    # costs: kernel JIT if not already cached, CUDA context warmup, etc.)
    scheduler._admit()
    for _ in range(WARMUP_STEPS):
        active = [s for s in scheduler.running if s.status.value == "running"]
        scheduler._decode_active(active)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        active = [s for s in scheduler.running if s.status.value == "running"]
        scheduler._decode_active(active)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = DECODE_STEPS * batch_size
    return total_tokens / elapsed


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    batch_sizes = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8, 16, 32]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_native(model_id, dtype=torch.bfloat16)

    print(f"\n{'batch_size':>10} | {'tokens/sec':>12} | {'ms/token (per-seq)':>20}")
    print("-" * 48)
    for batch_size in batch_sizes:
        throughput = benchmark_batch_size(model, tokenizer, batch_size)
        ms_per_token = 1000.0 / (throughput / batch_size)
        print(f"{batch_size:>10} | {throughput:>12.1f} | {ms_per_token:>20.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
