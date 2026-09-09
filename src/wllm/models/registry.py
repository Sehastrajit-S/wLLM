"""Architecture registry -- the same dispatch pattern real vLLM uses
(`vllm.model_executor.models.registry.ModelRegistry`), just at the scale this
codebase actually needs: map the HF config's `architectures[0]` string to the
model class that implements it, so `load_native` doesn't have to hardcode a
single class for every checkpoint.

Every entry below has a real model file backing it (llama.py, qwen2.py,
gemma.py) and a verification test comparing it against real HF weights --
see tests/test_llama_architecture.py and tests/test_gemma_architecture.py.

A checkpoint family was deliberately NOT added here despite looking
superficially Llama-shaped in its published docs (Yi, XVERSE, Baichuan2,
InternLM2, Orion): none of them are native `transformers` architectures --
their HF repos ship custom `trust_remote_code=True` modeling code instead of
a built-in class, so there's no official small test checkpoint to verify
against and no HF baseline to diff against without trusting arbitrary remote
code. Registering them on the strength of published architecture docs alone,
with no way to actually verify the config-parsing/weight-layout assumptions
against real weights, doesn't meet this project's bar (see the exact-match
tests every other architecture here has). Granite was also considered and
rejected for now: it *is* native to `transformers`, but its real config has
`embedding_multiplier`/`attention_multiplier`/`residual_multiplier`/
`logits_scaling` fields that are genuine architecture, not a config flag --
registering it as a bare LlamaForCausalLM would pass against a tiny test
checkpoint (where those all happen to be 1.0, a no-op) while being silently
wrong on every real Granite checkpoint.
"""
from __future__ import annotations

from wllm.models.gemma import GemmaForCausalLM
from wllm.models.llama import LlamaForCausalLM
from wllm.models.qwen2 import Qwen2ForCausalLM

ARCHITECTURE_REGISTRY: dict[str, type] = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    "GemmaForCausalLM": GemmaForCausalLM,
}


def resolve_model_class(architectures: list[str]) -> type:
    """`architectures` is config.json's own list (HF puts exactly one
    architecture there in practice, but the field is a list, so check all of
    them same as HF's own AutoModel dispatch does).
    """
    for arch in architectures:
        if arch in ARCHITECTURE_REGISTRY:
            return ARCHITECTURE_REGISTRY[arch]
    supported = ", ".join(sorted(ARCHITECTURE_REGISTRY))
    raise ValueError(
        f"Unsupported architecture(s) {architectures!r} -- wLLM currently implements: {supported}. "
        "See src/wllm/models/registry.py to add a new one."
    )
