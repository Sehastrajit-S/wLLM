"""Decode-only throughput for plain HuggingFace `transformers` inference --
no wLLM, no vLLM, just AutoModelForCausalLM.generate()'s underlying
incremental-decode mechanics (HF's own KV cache, no continuous batching, no
PagedAttention, no CUDA graphs). This is the "what do you get for free"
baseline the other two benchmark scripts (benchmark_cuda_graph.py,
vllm_bench_wsl.py) are measured against.

Same methodology as those: prefill once (untimed warmup), then time exactly
DECODE_STEPS single-token decode steps reusing the grown KV cache, for a
batch of `batch_size` identical prompts (repeated, not padded -- avoids
attention-mask bookkeeping being a confound, matching how the other two
scripts batch identical prompts too).

Run: python scripts/benchmark_baseline_hf.py [model_id] [comma,separated,batch,sizes]
"""
import sys
import time

sys.path.insert(0, "src")

import torch
from transformers import AutoTokenizer

from wllm.baseline.model import DEFAULT_MODEL_ID
from wllm.baseline.model import load as load_baseline

PROMPT = [{"role": "user", "content": "Tell me a short story about a robot who learns to paint."}]
DECODE_STEPS = 64
WARMUP_STEPS = 8


@torch.inference_mode()
def benchmark_batch_size(model, tokenizer, batch_size: int) -> float:
    prompt_ids = tokenizer.apply_chat_template(
        PROMPT, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"].to(model.device)
    input_ids = prompt_ids.repeat(batch_size, 1)

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)

    for _ in range(WARMUP_STEPS):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (DECODE_STEPS * batch_size) / elapsed


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    batch_sizes = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4, 16]

    print(f"Loading {model_id} via plain transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, _ = load_baseline(model_id, dtype=torch.bfloat16)

    print(f"\n{'batch_size':>10} | {'tokens/sec':>12}")
    print("-" * 28)
    for batch_size in batch_sizes:
        throughput = benchmark_batch_size(model, tokenizer, batch_size)
        print(f"{batch_size:>10} | {throughput:>12.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
