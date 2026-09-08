"""A Linear layer that stores its weight quantized (compact, in GPU memory)
and dequantizes on the fly each forward call. This is what actually delivers
the VRAM savings quantization is for -- converting the whole model to fp16
at load time would use the same steady-state memory as never quantizing at
all, just with a smaller download.

Correctness-first: dequantizing on every forward call is wasted repeated
work (the weight doesn't change between calls). Caching/fusing this into
the matmul is exactly the kind of thing Phase 3 (kernel/scheduler
performance) should revisit -- deferred deliberately, not forgotten.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from wllm.quant.dequant import dequantize


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        raw_data: torch.Tensor,
        ggml_type: int,
        out_features: int,
        in_features: int,
        bias: torch.Tensor | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.register_buffer("raw_data", raw_data, persistent=True)
        self.ggml_type = ggml_type
        self.out_features = out_features
        self.in_features = in_features
        self.compute_dtype = compute_dtype
        if bias is not None:
            self.register_buffer("bias", bias.to(compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = dequantize(self.raw_data, self.ggml_type, self.out_features * self.in_features)
        weight = flat.view(self.out_features, self.in_features).to(self.compute_dtype)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, ggml_type={self.ggml_type}"
