"""Decode-throughput benchmark for REAL vLLM, meant to run inside WSL2
(vLLM has no official Windows support -- this is the whole reason wLLM
exists -- but WSL2 gives genuine CUDA passthrough to the same physical GPU,
so this is an honest apples-to-apples comparison point).

Measures the same thing benchmark_cuda_graph.py measures for wLLM:
aggregate decode tokens/sec at a few fixed concurrent-request counts, using
vLLM's offline LLM API with a fixed max_tokens so every request does the
same fixed amount of decode work (mirrors the other script's methodology,
not a full request-arrival-pattern serving benchmark).

Run inside a WSL2 venv with vllm installed:
    python scripts/vllm_bench_wsl.py [model_id] [comma,separated,batch,sizes]
"""
import sys
import time

from vllm import LLM, SamplingParams

PROMPT = "Tell me a short story about a robot who learns to paint."
DECODE_TOKENS = 64


def _decode_only_throughput(outputs, batch_size: int) -> float | None:
    """Isolates decode time from prefill using vLLM's own per-request
    metrics (first_token_time -> finished_time), matching
    benchmark_cuda_graph.py's methodology of timing only the decode loop,
    not prompt processing. Returns None if metrics aren't populated (older/
    newer vLLM versions may name these differently) so the caller can fall
    back to a wall-clock measurement instead of silently reporting a wrong
    number.
    """
    try:
        decode_times = [o.metrics.finished_time - o.metrics.first_token_time for o in outputs]
        tokens_after_first = DECODE_TOKENS - 1
        total_tokens = tokens_after_first * batch_size
        total_decode_time = max(decode_times)  # batched: bounded by the slowest request
        return total_tokens / total_decode_time
    except (AttributeError, TypeError, ZeroDivisionError):
        return None


def benchmark_batch_size(llm: LLM, batch_size: int) -> tuple[float, bool]:
    """Returns (tokens/sec, is_decode_only) -- is_decode_only tells the
    caller whether prefill time was successfully excluded.
    """
    params = SamplingParams(temperature=0.0, max_tokens=DECODE_TOKENS, ignore_eos=True)
    prompts = [PROMPT] * batch_size

    # Warmup: pays one-time costs (CUDA context, kernel JIT/graph capture)
    llm.generate(prompts, params, use_tqdm=False)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    wall_elapsed = time.perf_counter() - t0

    decode_only = _decode_only_throughput(outputs, batch_size)
    if decode_only is not None:
        return decode_only, True

    # Fallback: wall-clock includes prefill, so this under-reports decode
    # throughput somewhat (flagged to the caller via is_decode_only=False).
    return (DECODE_TOKENS * batch_size) / wall_elapsed, False


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    batch_sizes = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4, 16]

    import os
    enforce_eager = os.environ.get("VLLM_BENCH_ENFORCE_EAGER") == "1"
    print(f"Loading {model_id} in vLLM (enforce_eager={enforce_eager})...")
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.85, enforce_eager=enforce_eager)

    print(f"\n{'batch_size':>10} | {'tokens/sec':>12} | note")
    print("-" * 48)
    for batch_size in batch_sizes:
        throughput, is_decode_only = benchmark_batch_size(llm, batch_size)
        note = "decode-only" if is_decode_only else "wall-clock (includes prefill, under-reports)"
        print(f"{batch_size:>10} | {throughput:>12.1f} | {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
