"""Same measurement as benchmark_decode.py, but through CUDAGraphDecoder
instead of eager Scheduler._decode_active, to quantify the actual CUDA
graph speedup at each bucketed batch size.

Run: python scripts/benchmark_cuda_graph.py [model_id] [comma,separated,batch,sizes] [--gguf PATH] [--tokenizer ID]
"""
import argparse
import sys
import time

sys.path.insert(0, "src")

import torch
from transformers import AutoTokenizer

from wllm.baseline.model import DEFAULT_MODEL_ID
from wllm.engine.cuda_graph_decoder import CUDAGraphDecoder
from wllm.engine.kv_cache import KVCacheManager
from wllm.models.qwen2 import load_native
from wllm.quant.gguf_loader import load_gguf

PROMPT = [{"role": "user", "content": "Tell me a short story about a robot who learns to paint."}]
DECODE_STEPS = 64
WARMUP_STEPS = 8
BUCKET_SIZES = (1, 2, 4, 8, 16, 32)


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


@torch.inference_mode()
def benchmark_batch_size(model, tokenizer, decoder: CUDAGraphDecoder, cache, batch_size: int) -> float:
    prompt_ids = tokenizer.apply_chat_template(
        PROMPT, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].tolist()
    input_ids = torch.tensor([prompt_ids], device="cuda")

    tokens = []
    for i in range(batch_size):
        seq_id = 1000 + i  # keep away from scratch's reserved id range
        model.prefill_with_cache(cache, seq_id, input_ids)
        tokens.append(100 + i)
    seq_ids = [1000 + i for i in range(batch_size)]

    for _ in range(WARMUP_STEPS):
        logits = decoder.decode(seq_ids, tokens)
        tokens = logits[:, -1, :].argmax(-1).tolist()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        logits = decoder.decode(seq_ids, tokens)
        tokens = logits[:, -1, :].argmax(-1).tolist()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (DECODE_STEPS * batch_size) / elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID)
    parser.add_argument("batch_sizes", nargs="?", default=None, help="comma,separated,batch,sizes")
    parser.add_argument("--gguf", default=None, help="Path to a single-file GGUF checkpoint (Q4_0/Q8_0) -- loads via load_gguf instead of load_native")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer repo for --gguf (defaults to model_id)")
    args = parser.parse_args()

    bucket_sizes = tuple(int(b) for b in args.batch_sizes.split(",")) if args.batch_sizes else BUCKET_SIZES

    print(f"Loading {args.gguf or args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_id)
    if args.gguf:
        model = load_gguf(args.gguf, device="cuda", compute_dtype=torch.bfloat16)
    else:
        model = load_native(args.model_id, dtype=torch.bfloat16)

    max_batch = max(bucket_sizes)
    cache = make_cache(model, num_blocks=max_batch * 32 + max_batch)  # +scratch headroom
    decoder = CUDAGraphDecoder(model, cache, bucket_sizes=bucket_sizes, max_blocks_per_seq=32)
    print("Capturing CUDA graphs...")
    decoder.capture()

    print(f"\n{'batch_size':>10} | {'tokens/sec':>12} | {'ms/token (per-seq)':>20}")
    print("-" * 48)
    for batch_size in bucket_sizes:
        throughput = benchmark_batch_size(model, tokenizer, decoder, cache, batch_size)
        ms_per_token = 1000.0 / (throughput / batch_size)
        print(f"{batch_size:>10} | {throughput:>12.1f} | {ms_per_token:>20.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
