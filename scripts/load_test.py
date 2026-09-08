"""Load-tests a running wLLM server: fires concurrent chat-completion
requests and reports throughput and latency percentiles (including
time-to-first-token, measured from the streaming response).

Run against a server already started separately, e.g.:
    python scripts/run_server.py --model Qwen/Qwen2.5-0.5B-Instruct --cuda-graphs
    python scripts/load_test.py --url http://localhost:8000 --concurrency 8 --requests 64
"""
import argparse
import asyncio
import statistics
import sys
import time

import httpx


async def _one_request(client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int) -> dict:
    start = time.monotonic()
    first_token_time = None
    chunk_count = 0

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk_count += 1
            if first_token_time is None:
                first_token_time = time.monotonic()

    end = time.monotonic()
    return {
        "total_latency_s": end - start,
        "ttft_s": (first_token_time - start) if first_token_time else None,
        "chunk_count": chunk_count,
    }


async def _worker(client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int, queue: asyncio.Queue, results: list) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            results.append(await _one_request(client, model, prompt, max_tokens))
        except Exception as e:
            results.append({"error": str(e)})


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


async def run(url: str, model: str | None, concurrency: int, num_requests: int, max_tokens: int, prompt: str) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(num_requests):
        queue.put_nowait(1)

    results: list[dict] = []
    start = time.monotonic()
    async with httpx.AsyncClient(base_url=url, timeout=120.0) as client:
        if model is None:
            # The server's `model` field also selects LoRA adapters (see
            # server.py's _resolve_lora_id) -- an unrecognized value is a
            # legitimate 404, not a routing failure, so discover the real
            # base model id rather than guessing one.
            models_resp = await client.get("/v1/models")
            models_resp.raise_for_status()
            model = models_resp.json()["data"][0]["id"]
            print(f"using model: {model}")
        workers = [asyncio.create_task(_worker(client, model, prompt, max_tokens, queue, results)) for _ in range(concurrency)]
        await asyncio.gather(*workers)
    elapsed = time.monotonic() - start

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    print(f"\n{num_requests} requests, concurrency={concurrency}, elapsed={elapsed:.2f}s")
    print(f"throughput: {len(ok) / elapsed:.2f} requests/sec")
    print(f"errors: {len(errors)}/{num_requests}")
    for e in errors[:5]:
        print(f"  {e['error']}")

    if ok:
        latencies = [r["total_latency_s"] for r in ok]
        ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
        print("\ntotal latency (s):")
        print(f"  p50={_percentile(latencies, 0.50):.3f}  p95={_percentile(latencies, 0.95):.3f}  p99={_percentile(latencies, 0.99):.3f}  mean={statistics.mean(latencies):.3f}")
        if ttfts:
            print("time to first token (s):")
            print(f"  p50={_percentile(ttfts, 0.50):.3f}  p95={_percentile(ttfts, 0.95):.3f}  p99={_percentile(ttfts, 0.99):.3f}  mean={statistics.mean(ttfts):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load-test a running wLLM server")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="model field to send (defaults to whatever /v1/models reports as the base model)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=32, help="Total requests to send across all workers")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain what a KV cache is in one paragraph.")
    args = parser.parse_args()

    asyncio.run(run(args.url, args.model, args.concurrency, args.requests, args.max_tokens, args.prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
