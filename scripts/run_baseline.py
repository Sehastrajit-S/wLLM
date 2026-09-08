"""P1.1 baseline harness: loads a small HF model and runs single-request
chat generation. This establishes the ground-truth reference for later phases.

Run: python scripts/run_baseline.py [model_id]
"""
import sys
import time

sys.path.insert(0, "src")

import torch

from wllm.baseline.generate import generate_text
from wllm.baseline.model import DEFAULT_MODEL_ID, load


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID

    print(f"Loading {model_id}...")
    t0 = time.time()
    model, tokenizer = load(model_id)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    messages = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]

    print("\nGenerating (greedy)...")
    t0 = time.time()
    text = generate_text(model, tokenizer, messages, max_new_tokens=64)
    elapsed = time.time() - t0

    print(f"\n--- Output ---\n{text}\n--------------")
    print(f"\n{elapsed:.2f}s elapsed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
