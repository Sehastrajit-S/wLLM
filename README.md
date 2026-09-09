<div align="center">
  <img src="https://raw.githubusercontent.com/Sehastrajit-S/wLLM/main/icons/wLLM.png" width="220" alt="wLLM logo">

  # wLLM

  **A production-grade LLM inference server, built from scratch, natively for Windows.**

  [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
  [![CI](https://github.com/Sehastrajit-S/wLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/Sehastrajit-S/wLLM/actions/workflows/ci.yml)
  [![PyPI](https://img.shields.io/pypi/v/wllm-server.svg)](https://pypi.org/project/wllm-server/)

  ```bash
  pip install wllm-server
  ```
</div>

---

vLLM doesn't run on Windows. wLLM is an independent reimplementation of vLLM's core ideas (PagedAttention, continuous batching, CUDA graphs) as a native CUDA/C++ + PyTorch inference engine that runs directly on Windows, no WSL2 or Docker required, with an OpenAI-compatible API on top.

Single GPU, single machine, NVIDIA CUDA only. Everything below was built and benchmarked on an RTX 3060.

## Why this exists

If you have a Windows machine with an NVIDIA GPU and want a real inference server (continuous batching, paged KV cache, CUDA graphs, quantization, guided decoding), your options have been "install WSL2 and hope it all works" or "don't." wLLM is the other option: a from-scratch engine that targets Windows as a first-class platform, with the same production concerns (auth, rate limiting, structured logging, Windows Service packaging) as any real deployment target.

## Features

**Core engine**
- Custom CUDA PagedAttention kernel (block-based KV cache, no wasted memory from over-allocation)
- Continuous batching scheduler with admission control
- CUDA graph capture for decode (3-7x throughput over the eager path)
- Automatic prefix caching (shared system prompts / multi-turn history reused, not recomputed)
- Chunked prefill (long prompts don't stall other requests' decode progress)
- CPU-swap preemption (evicted sequences resume exactly where they left off, no lost work)
- GGUF quantization (F32 / F16 / Q8_0 / Q4_0 / Q4_1 / Q6_K) with dequant-on-the-fly linear layers

**Serving**
- OpenAI-compatible REST API: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/models`
- Streaming (SSE) and non-streaming responses
- Guided decoding (JSON-schema-constrained generation)
- Tool / function calling (native `<tool_call>` format)
- Speculative decoding (n-gram prompt-lookup drafting, no draft model needed)
- Multi-LoRA hot-swap adapter serving, selected per-request via the `model` field
- Beam search
- Embedding and rerank endpoints

**Production**
- API-key auth and per-client rate limiting (both opt-in)
- Structured JSON logging
- Prometheus metrics (`/metrics`)
- Windows Service packaging (runs as a real background service, not just a terminal process)
- CI (GitHub Actions), lint-clean via `ruff`

## Supported models

**Qwen2 / Qwen2.5**, **Llama** (Llama 2, TinyLlama, and other checkpoints that don't rely on Llama 3's extended-context RoPE scaling, which isn't implemented yet), and **Gemma** (1). Architecture support is a registry (`src/wllm/models/registry.py`), the same dispatch pattern real vLLM uses: a checkpoint's `config.json` `architectures` field is looked up against a table of implemented model classes, the same way vLLM's own `ModelRegistry` does at much larger scale. Adding an architecture means a new model file plus a verification test that exact-matches its logits against real HF weights (see `tests/test_llama_architecture.py`, `tests/test_gemma_architecture.py`), not touching the PagedAttention kernel, scheduler, or KV cache -- those stay architecture-agnostic.

Qwen2 and Llama are close enough (RMSNorm, RoPE, grouped-query attention, SwiGLU MLP, identical HF weight naming -- Qwen2's checkpoint format was deliberately designed as a drop-in match for Llama's) that Qwen2 is implemented as a one-line subclass of Llama (`src/wllm/models/qwen2.py`), differing only in a config-driven qkv bias flag. Gemma (`src/wllm/models/gemma.py`) is the first architecture that's genuinely different rather than a config flag: a `(1 + weight)` RMSNorm parameterization, a GeGLU (tanh-approximate GELU) MLP instead of SwiGLU, embeddings scaled by `hidden_size**0.5` right after lookup, and a `head_dim` that isn't derivable from `hidden_size / num_attention_heads` the way every other architecture here has it -- all four wired in as real code, not assumptions, and it still gets the exact same PagedAttention/CUDA-graph decode path Llama and Qwen2 do, since attention itself is unchanged. Gemma 2/3 (sliding-window attention, attention/final logit softcapping, sandwich normalization) are deliberately not covered: softcapping has no equivalent in `torch.nn.functional.scaled_dot_product_attention` or the custom PagedAttention CUDA kernel, so supporting them for real means kernel work, not a new model file.

Weight format: HF safetensors (bf16/fp16, VRAM permitting) or a single-file GGUF checkpoint (F32/F16/Q8_0/Q4_0/Q4_1/Q6_K). Not yet supported: AWQ/GPTQ/bitsandbytes, split/multi-shard GGUF files, other K-quants (Q4_K/Q5_K/etc.) and I-quants, or any architecture outside the registry above (Mistral, Gemma 2+, Phi, Falcon, DeepSeek, etc. would each need their own model file and, in several of those cases, kernel changes too, the same way vLLM itself has one model file per architecture).

## Benchmarks

Three points of comparison, same RTX 3060, same prompt, same decode-only methodology: plain HuggingFace `transformers` (no serving engine at all, just `AutoModelForCausalLM` and its own KV cache), wLLM, and real vLLM 0.28.0. vLLM was run under WSL2, since it has no native Windows support at all, so this is the closest possible apples-to-apples comparison on identical hardware.

![Decode throughput: no serving engine vs wLLM vs vLLM](https://raw.githubusercontent.com/Sehastrajit-S/wLLM/main/benchmarks/throughput_chart.png)

Two things stand out. First, a real serving engine is a genuine, substantial win over doing nothing: wLLM is 1.3-3.8x faster than plain `transformers`, vLLM 1.6-6.0x, purely from PagedAttention, continuous batching, and CUDA graphs, same weights and same GPU either way. Second, that gain shrinks as the model grows, from roughly 4-6x at 0.5B down to about 1.3-1.8x at 3B: at small model sizes, per-step Python and kernel-launch overhead dominates, and that's exactly what a serving engine eliminates; at 3B, raw matmul compute is a bigger share of the total, so there's proportionally less overhead left to cut.

![Decode throughput table: no serving engine vs wLLM vs vLLM](https://raw.githubusercontent.com/Sehastrajit-S/wLLM/main/benchmarks/throughput_table.png)

wLLM captures most of the value vLLM adds over plain `transformers`, not just a small fraction of it: at 0.5B it gets roughly 64-77% of vLLM's speedup-over-baseline, and by 3B it's essentially matching vLLM's gain over baseline (85%+). The remaining wLLM-vs-vLLM gap (vLLM still leads by 1.2-1.6x outright, mostly from its more heavily-optimized attention kernels and `torch.compile` fusion) is a gap between two already-optimized systems, not "optimized vs. unoptimized."

### System configuration used

| Component | HF / wLLM (native Windows) | vLLM (WSL2) |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB) | same physical GPU, passed through |
| OS | Windows 11 Pro | Ubuntu 24.04.2 LTS (WSL2 kernel 5.15.167.4) |
| NVIDIA driver | 610.88 | same driver, shared via WSL2 GPU passthrough |
| CUDA Toolkit | 13.3 | CUDA 13.0 (bundled with vLLM's torch wheel) |
| PyTorch | 2.9.1+cu130 | 2.13.0+cu130 |
| transformers version | 5.16.1 (HF baseline only) | n/a |
| vLLM version | n/a | 0.28.0 |
| Compiler | MSVC 2022 + nvcc | n/a (Linux wheel, prebuilt) |
| Models | Qwen2.5-0.5B / 1.5B / 3B-Instruct, bf16 | same checkpoints, bf16 |

wLLM and vLLM were each run with their fastest available configuration: wLLM with CUDA graphs enabled, vLLM with `torch.compile` and CUDA graphs enabled (its default); the HF baseline uses no such optimizations, by design, since it's standing in for "no serving engine." See [`scripts/benchmark_baseline_hf.py`](scripts/benchmark_baseline_hf.py) (HF side), [`scripts/benchmark_cuda_graph.py`](scripts/benchmark_cuda_graph.py) (wLLM side), and [`scripts/vllm_bench_wsl.py`](scripts/vllm_bench_wsl.py) (vLLM side, run inside WSL2) to reproduce, and [`scripts/make_comparison_chart.py`](scripts/make_comparison_chart.py) to regenerate the chart and table above from the source numbers.

## Requirements

- Windows 10/11
- Python 3.11 or 3.12
- For GPU inference (the default, and the only way to get PagedAttention's real performance and CUDA graphs): an NVIDIA GPU (compute capability 7.0+; developed and tested on an RTX 3060), the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (matching your installed driver), and Visual Studio 2022 (or Build Tools) with the "Desktop development with C++" workload, for MSVC
- For CPU inference (`--device cpu`): none of the above -- see below

You don't need to manually configure `PATH`/`CUDA_HOME`/`vcvarsall.bat`. wLLM locates MSVC and CUDA automatically the first time it needs to build its kernel (see [`src/wllm/kernels/_msvc_env.py`](src/wllm/kernels/_msvc_env.py)).

### CPU inference

`--device cpu` runs the real engine (continuous batching, paged KV cache, prefix caching, LoRA, guided decoding, all of it) with no GPU, no CUDA Toolkit, and no MSVC -- the custom PagedAttention CUDA kernel is only ever JIT-compiled on the CUDA path; on CPU, decode instead runs a plain-PyTorch fallback with the same math (see [`src/wllm/kernels/paged_attention.py`](src/wllm/kernels/paged_attention.py)). This is a correctness fallback, not a performance target: no PagedAttention kernel speed and no CUDA graphs (`--cuda-graphs` is rejected together with `--device cpu`, since graph capture has no CPU equivalent at all). Useful for developing/testing wLLM on a machine without an NVIDIA GPU, not for serving real traffic.

## Installation

```bash
pip install wllm-server
# or: uv pip install wllm-server / uv add wllm-server
```

(The PyPI package is named `wllm-server` -- "wllm" alone was rejected as too similar to the real `vllm` package. `import wllm` and the `wllm-server` CLI command are unaffected.)

Or from source:

```bash
git clone https://github.com/Sehastrajit-S/wLLM.git
cd wLLM
pip install -e .
```

The first request that triggers a decode step will JIT-compile the CUDA kernel (via `torch.utils.cpp_extension`). This takes about a minute, once, and is cached afterward.

### Docker

```bash
docker build -t wllm .
docker run --gpus all -p 8000:8000 wllm --model Qwen/Qwen2.5-0.5B-Instruct
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On Windows this means Docker Desktop's WSL2 backend -- there's no native-Windows-container path to NVIDIA GPU compute passthrough at all, so this image reintroduces the WSL2 dependency the direct install above exists to avoid. Use it if you already have Docker + WSL2 GPU passthrough set up and want a container; use the direct install if avoiding WSL2 entirely is the point.

## Quickstart

```bash
wllm-server --model Qwen/Qwen2.5-0.5B-Instruct --cuda-graphs --prefix-caching
```

Then talk to it like any OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 32
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(resp.choices[0].message.content)
```

Run `wllm-server --help` for the full flag list (CUDA graphs, prefix caching, chunked prefill, CPU swap, speculative decoding, LoRA adapters, auth, rate limiting).

## What's not (yet) here

Being upfront about scope, same as the rest of this README:

- Multi-GPU / multi-node (single GPU, single machine only)
- Multi-modal / vision-language models
- AMD or non-NVIDIA GPUs, DirectML. `torch-directml` was evaluated and rejected for now: its latest release is pinned to an older PyTorch version than wLLM already depends on, so the two currently can't coexist in one environment regardless of engineering effort here.
- The PagedAttention kernel is warp-parallel but not yet using tensor cores or fp16 native arithmetic; see the benchmark numbers above for where that leaves it relative to vLLM

## Development

```bash
pip install -e ".[dev]"
python scripts/check_toolchain.py   # verify MSVC/nvcc/CUDA build toolchain
python scripts/run_tests.py         # full test suite, batched (see its docstring for why)
ruff check src/ tests/ scripts/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Collaborators

<a href="https://github.com/Sehastrajit-S">
  <img src="https://raw.githubusercontent.com/Sehastrajit-S/wLLM/main/icons/collabrators/sehas.jpg" width="80" style="border-radius:50%" alt="Sehastrajit">
</a>

[Sehastrajit](https://github.com/Sehastrajit-S)

## License

Apache 2.0, see [LICENSE](LICENSE).
