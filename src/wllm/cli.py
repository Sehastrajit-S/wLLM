"""wLLM's command-line entry point (installed as `wllm-server` via
pyproject.toml's [project.scripts], the same "pip install and get a real
command" pattern vLLM's own `vllm serve` uses).

Run: wllm-server --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
"""
from __future__ import annotations

import argparse
import sys

import torch
import uvicorn

from wllm.api.server import create_app
from wllm.baseline.model import DEFAULT_MODEL_ID


def main() -> int:
    parser = argparse.ArgumentParser(description="wLLM OpenAI-compatible server")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="cpu runs correctly but without the CUDA PagedAttention kernel's speed or CUDA graph support (incompatible with --cuda-graphs)",
    )
    parser.add_argument("--gguf", default=None, help="Path to a GGUF file -- loads quantized weights instead of --model's safetensors")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer repo to use with --gguf (defaults to --model)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-blocks", type=int, default=256, help="KV cache blocks (block_size tokens each)")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cuda-graphs", action="store_true", help="Replay bucketed CUDA graphs for decode (3.5-8x throughput)")
    parser.add_argument("--prefix-caching", action="store_true", help="Reuse cached KV blocks across requests sharing an exact prompt prefix")
    parser.add_argument("--chunked-prefill-tokens", type=int, default=None, help="Max prompt tokens processed per scheduler step (enables chunked prefill)")
    parser.add_argument("--cpu-swap", action="store_true", help="Swap preempted sequences' KV cache to host RAM instead of recomputing from scratch")
    parser.add_argument("--speculative-decoding", action="store_true", help="N-gram-drafted multi-token verification for eligible (greedy) sequences")
    parser.add_argument("--speculative-ngram-size", type=int, default=3)
    parser.add_argument("--speculative-max-draft-len", type=int, default=4)
    parser.add_argument(
        "--lora-modules",
        nargs="*",
        default=None,
        metavar="NAME=PATH",
        help="Preload LoRA adapters, e.g. --lora-modules my-adapter=/path/to/adapter",
    )
    parser.add_argument(
        "--api-key",
        dest="api_keys",
        nargs="*",
        default=None,
        metavar="KEY",
        help="Require Authorization: Bearer <key> on /v1/* requests, one of these keys (unset = auth disabled)",
    )
    parser.add_argument("--rate-limit-rpm", type=int, default=None, help="Max requests per minute per client (unset = unlimited)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if args.device == "cpu" and args.cuda_graphs:
        parser.error("--cuda-graphs requires --device cuda (CUDA graphs have no CPU equivalent)")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    lora_modules = dict(entry.split("=", 1) for entry in args.lora_modules) if args.lora_modules else None
    app = create_app(
        args.model,
        device=args.device,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        dtype=dtype,
        gguf_path=args.gguf,
        tokenizer_id=args.tokenizer,
        use_cuda_graphs=args.cuda_graphs,
        enable_prefix_caching=args.prefix_caching,
        max_prefill_tokens_per_step=args.chunked_prefill_tokens,
        enable_cpu_swap=args.cpu_swap,
        enable_speculative_decoding=args.speculative_decoding,
        speculative_ngram_size=args.speculative_ngram_size,
        speculative_max_draft_len=args.speculative_max_draft_len,
        lora_modules=lora_modules,
        api_keys=args.api_keys,
        rate_limit_per_minute=args.rate_limit_rpm,
        log_level=args.log_level,
    )
    # log_config=None: uvicorn's own startup otherwise calls
    # logging.config.dictConfig() with its default plain-text config,
    # clobbering the JSON logging create_app() just set up.
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
