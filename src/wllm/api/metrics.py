from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS_TOTAL = Counter("wllm_requests_total", "Total requests received", ["endpoint"])
TOKENS_GENERATED_TOTAL = Counter("wllm_tokens_generated_total", "Total completion tokens generated")
TIME_TO_FIRST_TOKEN_SECONDS = Histogram("wllm_time_to_first_token_seconds", "Time to first generated token")
REQUEST_LATENCY_SECONDS = Histogram("wllm_request_latency_seconds", "Total request latency", ["endpoint"])

__all__ = [
    "CONTENT_TYPE_LATEST",
    "REQUESTS_TOTAL",
    "REQUEST_LATENCY_SECONDS",
    "TIME_TO_FIRST_TOKEN_SECONDS",
    "TOKENS_GENERATED_TOTAL",
    "generate_latest",
]
