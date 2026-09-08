"""P1.0 toolchain validation: compiles a trivial custom CUDA kernel via MSVC/nvcc
through torch's cpp_extension and checks the result against PyTorch's own add.

Run: python scripts/check_toolchain.py
"""
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wllm.kernels._msvc_env import ensure_msvc_on_path  # noqa: E402

CSRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "wllm", "kernels", "csrc")


def main() -> int:
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available to PyTorch. Check driver/torch CUDA wheel.")
        return 1

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"compute capability: {torch.cuda.get_device_capability(0)}")

    print("\nCompiling custom CUDA extension via MSVC/nvcc (first run only, may take a minute)...")
    t0 = time.time()
    try:
        ensure_msvc_on_path()
        ext = load(
            name="wllm_toolchain_check",
            sources=[os.path.join(CSRC_DIR, "vector_add.cu")],
            extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
            verbose=True,
        )
    except Exception as e:
        print(f"\nERROR: extension build failed: {e}")
        return 1
    print(f"Build succeeded in {time.time() - t0:.1f}s")

    a = torch.randn(1 << 20, device="cuda")
    b = torch.randn(1 << 20, device="cuda")

    out = ext.vector_add(a, b)
    expected = a + b

    max_err = (out - expected).abs().max().item()
    print(f"\nmax error vs torch reference: {max_err:.2e}")

    if max_err > 1e-5:
        print("ERROR: custom kernel output does not match reference.")
        return 1

    print("\nSUCCESS: MSVC + nvcc + PyTorch CUDA extension toolchain is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
