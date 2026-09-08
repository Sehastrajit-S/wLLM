import json
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
    torch.cuda.empty_cache()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"][0]["id"] == MODEL_ID


def test_metrics_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "wllm_requests_total" in resp.text


def test_chat_completions_non_streaming(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    print(f"\nnon-streaming: {text!r}")
    assert "Paris" in text
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_chat_completions_streaming(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    text = "".join(p["choices"][0]["delta"].get("content") or "" for p in payloads)
    print(f"\nstreaming reassembled: {text!r}")
    assert "Paris" in text

    finish_reasons = [p["choices"][0].get("finish_reason") for p in payloads]
    assert finish_reasons[-1] == "length"
    assert all(r is None for r in finish_reasons[:-1])


def test_legacy_completions(client):
    resp = client.post(
        "/v1/completions",
        json={"model": MODEL_ID, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    print(f"\nlegacy completion: {body['choices'][0]['text']!r}")
    assert "Paris" in body["choices"][0]["text"]


def test_embeddings_single_input(client):
    resp = client.post("/v1/embeddings", json={"model": MODEL_ID, "input": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    vec = body["data"][0]["embedding"]
    assert len(vec) > 0
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3
    assert body["usage"]["prompt_tokens"] > 0


def test_embeddings_batch_input_preserves_order(client):
    resp = client.post(
        "/v1/embeddings",
        json={"model": MODEL_ID, "input": ["first text", "second text", "third text"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 3
    assert [d["index"] for d in body["data"]] == [0, 1, 2]


def test_rerank_orders_documents_by_relevance(client):
    resp = client.post(
        "/v1/rerank",
        json={
            "model": MODEL_ID,
            "query": "What is the capital of France?",
            "documents": [
                "I like to bake chocolate cake on weekends.",
                "Paris is the capital city of France.",
                "The weather today is sunny and warm.",
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    print(f"\nrerank results: {body['results']}")
    assert len(body["results"]) == 3
    # the Paris document (originally index 1) should rank first
    assert body["results"][0]["index"] == 1
    scores = [r["relevance_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_respects_top_n(client):
    resp = client.post(
        "/v1/rerank",
        json={
            "model": MODEL_ID,
            "query": "capital of France",
            "documents": ["Paris is in France.", "Berlin is in Germany.", "Tokyo is in Japan."],
            "top_n": 2,
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


@pytest.fixture(scope="module")
def gguf_client():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from huggingface_hub import hf_hub_download

    gguf_path = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_0.gguf")
    app = create_app(MODEL_ID, num_blocks=64, block_size=16, gguf_path=gguf_path)
    with TestClient(app) as c:
        yield c
    torch.cuda.empty_cache()


def test_chat_completions_with_gguf_backend(gguf_client):
    resp = gguf_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"\nGGUF-backed server: {text!r}")
    assert "Paris" in text


@pytest.fixture(scope="module")
def cuda_graph_client():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    app = create_app(
        MODEL_ID, num_blocks=64, block_size=16, dtype=torch.float32, use_cuda_graphs=True, cuda_graph_bucket_sizes=(1, 2, 4)
    )
    with TestClient(app) as c:
        yield c
    torch.cuda.empty_cache()


def test_chat_completions_with_cuda_graphs_enabled(cuda_graph_client):
    resp = cuda_graph_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"\nCUDA-graph server: {text!r}")
    assert "Paris" in text


@pytest.fixture(scope="module")
def prefix_cache_client():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    app = create_app(MODEL_ID, num_blocks=64, block_size=16, dtype=torch.float32, enable_prefix_caching=True)
    with TestClient(app) as c:
        yield c
    torch.cuda.empty_cache()


def test_two_requests_sharing_a_system_prompt_both_answer_correctly(prefix_cache_client):
    shared_prefix = (
        "You are a helpful assistant with extensive knowledge of world geography and history. "
        "Always answer concisely in a single sentence."
    )
    for question, expected in [
        (f"{shared_prefix} What is the capital of France?", "Paris"),
        (f"{shared_prefix} What is the capital of Germany?", "Berlin"),
    ]:
        resp = prefix_cache_client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 16,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        text = resp.json()["choices"][0]["message"]["content"]
        print(f"\nprefix-cache server [{question[-40:]}]: {text!r}")
        assert expected in text


@pytest.fixture(scope="module")
def chunked_prefill_client():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    app = create_app(MODEL_ID, num_blocks=64, block_size=16, dtype=torch.float32, max_prefill_tokens_per_step=8)
    with TestClient(app) as c:
        yield c
    torch.cuda.empty_cache()


def test_chat_completions_with_chunked_prefill_enabled(chunked_prefill_client):
    resp = chunked_prefill_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"\nchunked-prefill server: {text!r}")
    assert "Paris" in text


@pytest.fixture(scope="module")
def cpu_swap_client():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    app = create_app(MODEL_ID, num_blocks=64, block_size=16, dtype=torch.float32, enable_cpu_swap=True)
    with TestClient(app) as c:
        yield c
    torch.cuda.empty_cache()


def test_chat_completions_with_cpu_swap_enabled(cpu_swap_client):
    """Real contention/actual-swap-triggering is validated at the scheduler
    level (test_cpu_swap.py); this just proves the flag is correctly wired
    through create_app -> AsyncEngine -> Scheduler for a real HTTP request.
    """
    resp = cpu_swap_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"\ncpu-swap server: {text!r}")
    assert "Paris" in text


@pytest.fixture(scope="module")
def speculative_client():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    app = create_app(MODEL_ID, num_blocks=64, block_size=16, dtype=torch.float32, enable_speculative_decoding=True)
    with TestClient(app) as c:
        yield c
    torch.cuda.empty_cache()


def test_chat_completions_with_speculative_decoding_streaming(speculative_client):
    """Uses a repetitive prompt so the n-gram drafter actually has something
    to find (real correctness/exact-match is validated at the scheduler
    level in test_speculative_decoding.py) -- this specifically exercises
    the streaming path, since multi-token-per-step acceptance is exactly
    what the AsyncEngine._on_token fix (report every new token, not just the
    latest) was needed for.
    """
    with speculative_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": "Repeat the following sentence exactly three times: The quick brown fox jumps over the lazy dog.",
                }
            ],
            "max_tokens": 40,
            "temperature": 0.0,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    text = "".join(p["choices"][0]["delta"].get("content") or "" for p in payloads)
    print(f"\nspeculative streaming: {text!r}")
    assert text.lower().count("quick brown fox") >= 2


def test_chat_completions_with_guided_json(client):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Generate a JSON object for the city of Paris, France."}],
            "max_tokens": 40,
            "temperature": 0.0,
            "guided_json": schema,
        },
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"\nguided-json server: {text!r}")
    parsed = json.loads(text)
    assert set(parsed.keys()) == {"city", "country"}


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}


def test_chat_completions_with_tool_calling_non_streaming(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "max_tokens": 60,
            "temperature": 0.0,
            "tools": [WEATHER_TOOL],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    choice = body["choices"][0]
    print(f"\ntool-call response: {choice!r}")
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert "paris" in args["location"].lower()
    # generation must have stopped right at </tool_call>, not rambled on
    assert "</tool_call>" not in (choice["message"]["content"] or "")


def test_chat_completions_with_tool_calling_streaming(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "max_tokens": 60,
            "temperature": 0.0,
            "tools": [WEATHER_TOOL],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    final_choice = payloads[-1]["choices"][0]
    print(f"\nstreaming tool-call final chunk: {final_choice!r}")
    assert final_choice["finish_reason"] == "tool_calls"
    tool_calls = final_choice["delta"]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "get_weather"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert "paris" in args["location"].lower()


def test_chat_completions_without_tools_behaves_normally(client):
    """Sanity: the tool-calling plumbing must not affect ordinary requests."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["message"]["tool_calls"] is None
    assert "Paris" in choice["message"]["content"]
    assert choice["finish_reason"] != "tool_calls"


def test_stop_string_truncates_output_and_cancels_early(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stop": ["."],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    print(f"\nstop-string truncated: {text!r}")
    assert "." not in text
    assert "Paris" in text
    assert body["choices"][0]["finish_reason"] == "stop"
