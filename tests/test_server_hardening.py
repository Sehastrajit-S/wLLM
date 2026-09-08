"""API-key auth, rate limiting, and structured JSON logging middleware, each
opt-in on create_app -- verified enabled and (separately) verified to have
zero effect when not configured, matching every other opt-in feature in this
codebase.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from fastapi.testclient import TestClient

from wllm.api.server import create_app

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _make_client(**kwargs):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    app = create_app(MODEL_ID, num_blocks=64, block_size=16, dtype=torch.float32, **kwargs)
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_client():
    with _make_client(api_keys=["secret-key-1", "secret-key-2"]) as c:
        yield c


def test_missing_api_key_is_rejected(auth_client):
    resp = auth_client.get("/v1/models")
    assert resp.status_code == 401


def test_wrong_api_key_is_rejected(auth_client):
    resp = auth_client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


def test_valid_api_key_is_accepted(auth_client):
    resp = auth_client.get("/v1/models", headers={"Authorization": "Bearer secret-key-1"})
    assert resp.status_code == 200


def test_second_configured_key_also_works(auth_client):
    resp = auth_client.get("/v1/models", headers={"Authorization": "Bearer secret-key-2"})
    assert resp.status_code == 200


def test_health_and_metrics_are_never_gated(auth_client):
    assert auth_client.get("/health").status_code == 200
    assert auth_client.get("/metrics").status_code == 200


def test_auth_disabled_by_default():
    with _make_client() as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200


@pytest.fixture(scope="module")
def rate_limited_client():
    with _make_client(rate_limit_per_minute=3) as c:
        yield c


def test_requests_within_limit_succeed(rate_limited_client):
    for _ in range(3):
        resp = rate_limited_client.get("/v1/models")
        assert resp.status_code == 200


def test_request_over_limit_is_rejected(rate_limited_client):
    resp = rate_limited_client.get("/v1/models")
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_health_is_not_rate_limited(rate_limited_client):
    # The fixture's budget is already exhausted by the tests above -- /health
    # must still succeed since only /v1/* is gated.
    resp = rate_limited_client.get("/health")
    assert resp.status_code == 200


def test_rate_limit_disabled_by_default():
    with _make_client() as client:
        for _ in range(10):
            assert client.get("/v1/models").status_code == 200


def test_structured_logging_emits_json_access_log():
    """configure_logging() deliberately replaces the ROOT logger's handlers
    wholesale (see its docstring) so a real deployment's uvicorn/access logs
    all come out as JSON -- which also removes pytest's caplog capture
    handler from root. Attaching a collecting handler directly to the
    "wllm" logger sidesteps that (a logger's own handlers still fire
    regardless of what its ancestors' handlers are), without weakening what
    configure_logging does for real use.
    """
    import logging

    records: list[logging.LogRecord] = []

    class _CollectingHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    wllm_logger = logging.getLogger("wllm")
    wllm_logger.addHandler(_CollectingHandler())
    try:
        with _make_client() as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 200
    finally:
        wllm_logger.handlers = [h for h in wllm_logger.handlers if not isinstance(h, _CollectingHandler)]

    assert records, "expected at least one access-log record"
    record = records[-1]
    extra = getattr(record, "extra_fields", None)
    assert extra is not None
    assert extra["method"] == "GET"
    assert extra["path"] == "/v1/models"
    assert extra["status"] == 200
    assert "latency_ms" in extra


def test_json_formatter_produces_valid_json():
    import logging

    from wllm.api.logging_config import JSONFormatter

    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="wllm", level=logging.INFO, pathname=__file__, lineno=1, msg="request", args=(), exc_info=None,
    )
    record.extra_fields = {"method": "GET", "path": "/v1/models", "status": 200, "latency_ms": 1.23}
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "request"
    assert payload["method"] == "GET"
    assert payload["status"] == 200
