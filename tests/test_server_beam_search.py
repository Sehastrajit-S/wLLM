"""HTTP-level beam search: use_beam_search/best_of/length_penalty on the
real /v1/chat/completions and /v1/completions endpoints, plus the documented
rejections (streaming, guided_json, tools combined with beam search).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from fastapi.testclient import TestClient

from wllm.api.server import create_app

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def client():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    app = create_app(MODEL_ID, num_blocks=64, block_size=16, dtype=torch.float32)
    with TestClient(app) as c:
        yield c


def test_chat_completions_beam_search_finds_the_answer(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 16,
            "use_beam_search": True,
            "best_of": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "Paris" in body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_completions_beam_search_finds_the_answer(client):
    resp = client.post(
        "/v1/completions",
        json={
            "model": MODEL_ID,
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "use_beam_search": True,
            "best_of": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    assert "Paris" in resp.json()["choices"][0]["text"]


def test_beam_search_rejects_streaming(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "use_beam_search": True,
            "stream": True,
        },
    )
    assert resp.status_code == 400


def test_beam_search_rejects_guided_json(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "use_beam_search": True,
            "guided_json": {"type": "object", "properties": {}},
        },
    )
    assert resp.status_code == 400


def test_beam_search_rejects_tools(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "use_beam_search": True,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        },
    )
    assert resp.status_code == 400
