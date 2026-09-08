"""HTTP-level LoRA tests: load/unload/list adapters via the management
endpoints, then select one through a real chat completion request the same
way an OpenAI client would -- by putting its name in the `model` field.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from wllm.api.server import create_app

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Qwen2.5-0.5B-Instruct's real dims (hidden_size=896, num_attention_heads=14,
# head_dim=64 -> num_heads*head_dim == hidden_size for this model), needed to
# build an o_proj adapter with shapes the real model will actually accept.
HIDDEN_SIZE = 896
RANK = 4


def _make_adapter_dir(tmp_dir: Path, seed: int) -> str:
    adapter_dir = tmp_dir / f"adapter_{seed}"
    adapter_dir.mkdir()
    config = {"r": RANK, "lora_alpha": RANK * 50.0, "target_modules": ["o_proj"]}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

    g = torch.Generator().manual_seed(seed)
    A = torch.randn(RANK, HIDDEN_SIZE, generator=g)
    B = torch.randn(HIDDEN_SIZE, RANK, generator=g)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight": A,
            "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight": B,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    return str(adapter_dir)


@pytest.fixture(scope="module")
def adapter_dirs():
    with tempfile.TemporaryDirectory() as d:
        tmp_dir = Path(d)
        yield {
            "preloaded-adapter": _make_adapter_dir(tmp_dir, seed=10),
            "dynamic-adapter": _make_adapter_dir(tmp_dir, seed=20),
        }


@pytest.fixture(scope="module")
def client(adapter_dirs):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    app = create_app(
        MODEL_ID,
        num_blocks=64,
        block_size=16,
        dtype=torch.float32,
        lora_modules={"preloaded-adapter": adapter_dirs["preloaded-adapter"]},
    )
    with TestClient(app) as c:
        yield c


def _chat(client, model: str, max_tokens: int = 12):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["choices"][0]["message"]["content"]


def test_preloaded_adapter_appears_in_model_list(client):
    resp = client.get("/v1/models")
    body = resp.json()
    ids = {m["id"]: m for m in body["data"]}
    assert "preloaded-adapter" in ids
    assert ids["preloaded-adapter"]["parent"] == MODEL_ID
    assert ids[MODEL_ID]["parent"] is None


def test_preloaded_adapter_changes_generation_vs_base_model(client):
    base_text = _chat(client, MODEL_ID)
    adapter_text = _chat(client, "preloaded-adapter")
    assert adapter_text != base_text


def test_unknown_model_name_returns_404(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "not-a-real-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert resp.status_code == 404


def test_dynamic_load_use_and_unload_adapter(client, adapter_dirs):
    load_resp = client.post(
        "/v1/load_lora_adapter", json={"lora_name": "dynamic-adapter", "lora_path": adapter_dirs["dynamic-adapter"]}
    )
    assert load_resp.status_code == 200

    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert "dynamic-adapter" in ids

    base_text = _chat(client, MODEL_ID)
    adapter_text = _chat(client, "dynamic-adapter")
    assert adapter_text != base_text

    unload_resp = client.post("/v1/unload_lora_adapter", json={"lora_name": "dynamic-adapter"})
    assert unload_resp.status_code == 200

    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert "dynamic-adapter" not in ids

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "dynamic-adapter", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert resp.status_code == 404
