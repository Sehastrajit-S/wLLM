"""OpenAI-compatible request/response schemas (subset). Fields beyond the
official OpenAI spec (top_k, min_p, repetition_penalty, guided_json) are
vLLM-style extensions -- harmless to include since OpenAI clients simply
omit them.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string, per the OpenAI spec (not a nested object)


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    guided_json: dict[str, Any] | str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    use_beam_search: bool = False
    best_of: int | None = None  # number of beams when use_beam_search is set (vLLM's legacy naming)
    length_penalty: float = 1.0


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    guided_json: dict[str, Any] | str | None = None
    use_beam_search: bool = False
    best_of: int | None = None
    length_penalty: float = 1.0


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class CompletionChunkChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str | None = None


class CompletionChunk(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "wllm"
    parent: str | None = None  # base model id this entry is derived from, set for LoRA adapters


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class LoadLoRARequest(BaseModel):
    lora_name: str
    lora_path: str


class UnloadLoRARequest(BaseModel):
    lora_name: str
