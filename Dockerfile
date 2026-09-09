# GPU image. Requires Docker Desktop's WSL2 backend + the NVIDIA Container
# Toolkit on Windows: there is no native-Windows-container path to NVIDIA
# GPU *compute* passthrough at all (WDDM GPU sharing for Windows containers
# doesn't cover CUDA), so `docker run --gpus all` on Windows means this image
# runs as a Linux container through WSL2 -- the same dependency the direct
# (non-Docker) install on this repo exists specifically to avoid. Use this
# image if you already have Docker + WSL2 GPU passthrough set up and want a
# container instead of a bare-metal install; use the direct install (see
# README's Installation section) if avoiding WSL2 entirely is the point.
#
# Base image is CUDA 13.0's devel variant (nvcc + headers), not the smaller
# runtime-only variant: the custom PagedAttention kernel is JIT-compiled via
# torch.utils.cpp_extension, which needs nvcc present, not just the CUDA
# runtime libraries.
FROM nvidia/cuda:13.0.3-devel-ubuntu22.04

# Ubuntu 22.04 ships Python 3.10; pyproject.toml requires 3.11 or 3.12.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/

# The CUDA-enabled wheel (not PyPI's default CPU-only torch) has to be
# installed explicitly, matching this image's CUDA 13.0 toolkit.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir -e .

# Compiles the CUDA kernel now (nvcc only needs the toolkit, not a live GPU,
# so this works fine at build time with no --gpus flag) instead of on the
# first real request, so a freshly-started container is immediately fast
# rather than paying that ~1-minute JIT cost on its first decode step.
RUN python -c "from wllm.kernels.paged_attention import _get_ext; _get_ext()"

EXPOSE 8000
ENTRYPOINT ["wllm-server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
