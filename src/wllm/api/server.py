from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from transformers import AutoTokenizer

from wllm.api.auth import APIKeyAuthMiddleware
from wllm.api.logging_config import AccessLogMiddleware, configure_logging
from wllm.api.metrics import (
    CONTENT_TYPE_LATEST,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    TIME_TO_FIRST_TOKEN_SECONDS,
    TOKENS_GENERATED_TOTAL,
    generate_latest,
)
from wllm.api.rate_limit import RateLimitMiddleware
from wllm.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    LoadLoRARequest,
    ModelCard,
    ModelList,
    RerankRequest,
    RerankResponse,
    RerankResult,
    UnloadLoRARequest,
    UsageInfo,
)
from wllm.api.stop_strings import StopStringFilter
from wllm.api.tool_calls import parse_tool_calls
from wllm.engine.async_engine import AsyncEngine
from wllm.engine.cuda_graph_decoder import CUDAGraphDecoder
from wllm.engine.embeddings import cosine_similarity, embed_text
from wllm.engine.guided_decoding import JSONSchemaGuide
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.lora import LoRARegistry
from wllm.engine.sampling import SamplingParams
from wllm.models.qwen2 import load_native
from wllm.quant.gguf_loader import load_gguf


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _sampling_params_from(req) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        min_p=req.min_p,
        repetition_penalty=req.repetition_penalty,
        # Note: `seed` is accepted but not yet honored -- torch.multinomial
        # takes one generator per call, and a batched decode step samples
        # multiple sequences (each with their own seed) in a single call.
        # Wiring true per-request determinism through that needs either a
        # per-row sampling loop or an upstream torch feature; deferred
        # rather than faking it with a shared/racy global seed.
    )


def _build_guide(req, engine: AsyncEngine, vocab_tokenizer_name: str) -> JSONSchemaGuide | None:
    if req.guided_json is None:
        return None
    return JSONSchemaGuide(vocab_tokenizer_name, req.guided_json, vocab_size=engine.model.cfg.vocab_size)


def _reject_unsupported_beam_search_combo(req) -> None:
    """Beam search's winning sequence isn't known until the whole group
    finishes (see AsyncEngine.generate_beam_search), which rules out
    streaming outright, and BeamGroup has no token-mask/tool-call-parsing
    support (see beam_search.py) -- reject rather than silently ignoring
    those request fields.
    """
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=True is not supported with use_beam_search")
    if getattr(req, "guided_json", None) is not None:
        raise HTTPException(status_code=400, detail="guided_json is not supported with use_beam_search")
    if getattr(req, "tools", None):
        raise HTTPException(status_code=400, detail="tools is not supported with use_beam_search")


def _resolve_lora_id(requested_model: str, state: ServerState, engine: AsyncEngine) -> str | None:
    """OpenAI clients pick an adapter the same way they pick a model: by
    putting its name in the request's `model` field (vLLM's own convention,
    reused here rather than adding a bespoke `lora_id` field). The base
    model's own id always means "no adapter".
    """
    if requested_model == state.model_id:
        return None
    if requested_model in engine.lora_registry.list_adapters():
        return requested_model
    raise HTTPException(status_code=404, detail=f"model '{requested_model}' not found (not the base model or a loaded LoRA adapter)")


class ServerState:
    engine: AsyncEngine | None = None
    model_id: str = ""


def create_app(
    model_id: str,
    device: str = "cuda",
    num_blocks: int = 256,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    gguf_path: str | None = None,
    tokenizer_id: str | None = None,
    use_cuda_graphs: bool = False,
    cuda_graph_bucket_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    cuda_graph_max_blocks_per_seq: int = 256,
    enable_prefix_caching: bool = False,
    max_prefill_tokens_per_step: int | None = None,
    enable_cpu_swap: bool = False,
    enable_speculative_decoding: bool = False,
    speculative_ngram_size: int = 3,
    speculative_max_draft_len: int = 4,
    lora_modules: dict[str, str] | None = None,
    api_keys: list[str] | None = None,
    rate_limit_per_minute: int | None = None,
    log_level: str = "INFO",
) -> FastAPI:
    """If `gguf_path` is given, model weights load from that GGUF file
    (quantized layers stay compact in GPU memory, see quant/gguf_loader.py)
    instead of the safetensors checkpoint at `model_id`. GGUF embeds its own
    tokenizer, but we deliberately use the HF tokenizer instead (see
    gguf_loader's module docstring) -- `tokenizer_id` picks which one,
    defaulting to `model_id` when not given.

    `use_cuda_graphs` replays bucketed CUDA graphs for the decode step
    instead of the eager per-op path -- measured 3.5-8x decode throughput
    depending on batch size (see cuda_graph_decoder.py). `num_blocks` is
    bumped automatically by `max(cuda_graph_bucket_sizes) - 1` to cover the
    permanent scratch sequences CUDAGraphDecoder needs for padding, so the
    caller doesn't need to account for that headroom themselves.

    `enable_prefix_caching` reuses already-cached KV blocks across requests
    that share an exact token prefix (e.g. a common system prompt, or
    multi-turn chat history) -- skips recomputing and rewriting that shared
    portion entirely rather than just avoiding a redundant copy.

    `max_prefill_tokens_per_step`, if given, spreads a long prompt's prefill
    across multiple scheduler steps (chunked prefill) so it can't stall
    other sequences' decode progress for more than one step's worth of time.

    `enable_cpu_swap` makes preemption under cache pressure swap a victim's
    KV cache to host RAM and resume it later instead of discarding its
    progress and recomputing from scratch.

    `enable_speculative_decoding` drafts multiple tokens at once via n-gram
    lookup against a sequence's own history and verifies them in one forward
    pass -- exact (not approximate) for eligible sequences (greedy,
    repetition_penalty disabled, no guided-decoding constraint active; see
    scheduler.py's module docstring for why), falling back to ordinary
    single-token decode for everything else or whenever nothing repeats.

    `lora_modules`, if given, maps adapter name -> PEFT adapter directory
    path, preloaded at startup (vLLM's --lora-modules convention). More
    adapters can be loaded/unloaded later at runtime via
    /v1/load_lora_adapter and /v1/unload_lora_adapter. A request selects an
    adapter the same way it selects a model: by putting the adapter's name
    in the request's `model` field.

    `api_keys`, if given, requires every /v1/* request to present one of
    these keys via `Authorization: Bearer <key>` (disabled entirely, no
    performance/behavior cost, when not given). `rate_limit_per_minute`, if
    given, caps requests per client (by API key if auth is enabled, else by
    IP) in a rolling 60s window. `log_level` controls the structured JSON
    logging every request is recorded through -- see logging_config.py.
    """
    configure_logging(log_level)
    state = ServerState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
        if gguf_path:
            model = load_gguf(gguf_path, device=device, compute_dtype=dtype)
        else:
            model = load_native(model_id, device=device, dtype=dtype)

        effective_num_blocks = num_blocks
        if use_cuda_graphs:
            effective_num_blocks += max(cuda_graph_bucket_sizes) - 1
        cache = KVCacheManager(
            num_layers=model.cfg.num_hidden_layers,
            num_blocks=effective_num_blocks,
            block_size=block_size,
            num_kv_heads=model.cfg.num_key_value_heads,
            head_dim=model.cfg.head_dim,
            device=device,
            dtype=torch.float32,
            enable_prefix_caching=enable_prefix_caching,
        )

        graph_decoder = None
        if use_cuda_graphs:
            graph_decoder = CUDAGraphDecoder(
                model, cache, bucket_sizes=cuda_graph_bucket_sizes, max_blocks_per_seq=cuda_graph_max_blocks_per_seq
            )
            graph_decoder.capture()

        lora_registry = LoRARegistry()
        for adapter_name, adapter_path in (lora_modules or {}).items():
            lora_registry.load_adapter(adapter_name, adapter_path, device=device, dtype=dtype)
        engine = AsyncEngine(
            model,
            cache,
            tokenizer,
            device=device,
            graph_decoder=graph_decoder,
            max_prefill_tokens_per_step=max_prefill_tokens_per_step,
            enable_cpu_swap=enable_cpu_swap,
            enable_speculative_decoding=enable_speculative_decoding,
            speculative_ngram_size=speculative_ngram_size,
            speculative_max_draft_len=speculative_max_draft_len,
            lora_registry=lora_registry,
        )
        engine.start()
        state.engine = engine
        state.model_id = model_id
        try:
            yield
        finally:
            await engine.stop()
            state.engine = None

    app = FastAPI(title="wLLM", lifespan=lifespan)

    # Order matters: add_middleware's first call ends up outermost, so
    # AccessLog (added first) sees every request/response including ones
    # Auth or RateLimit reject; Auth (added second) runs before RateLimit so
    # unauthenticated noise doesn't consume real clients' quota.
    app.add_middleware(AccessLogMiddleware)
    if api_keys:
        app.add_middleware(APIKeyAuthMiddleware, valid_keys=frozenset(api_keys))
    if rate_limit_per_minute is not None:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit_per_minute)

    def _engine() -> AsyncEngine:
        if state.engine is None:
            raise HTTPException(status_code=503, detail="engine not ready")
        return state.engine

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        engine = _engine()
        created = int(time.time())
        cards = [ModelCard(id=state.model_id, created=created)]
        cards += [
            ModelCard(id=adapter_id, created=created, parent=state.model_id)
            for adapter_id in engine.lora_registry.list_adapters()
        ]
        return ModelList(data=cards)

    @app.post("/v1/load_lora_adapter")
    async def load_lora_adapter(req: LoadLoRARequest):
        engine = _engine()
        engine.lora_registry.load_adapter(req.lora_name, req.lora_path, device=device, dtype=dtype)
        return {"status": "success", "lora_name": req.lora_name}

    @app.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(req: UnloadLoRARequest):
        engine = _engine()
        engine.lora_registry.unload_adapter(req.lora_name)
        return {"status": "success", "lora_name": req.lora_name}

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
        REQUESTS_TOTAL.labels(endpoint="embeddings").inc()
        engine = _engine()
        texts = [req.input] if isinstance(req.input, str) else req.input

        prompt_tokens = 0
        data = []
        for i, text in enumerate(texts):
            prompt_tokens += len(engine.tokenizer.encode(text, add_special_tokens=False))
            vec = embed_text(engine.model, engine.tokenizer, text, device=device)
            data.append(EmbeddingData(index=i, embedding=vec.tolist()))

        return EmbeddingResponse(
            data=data,
            model=req.model,
            usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens),
        )

    @app.post("/v1/rerank")
    async def rerank(req: RerankRequest) -> RerankResponse:
        REQUESTS_TOTAL.labels(endpoint="rerank").inc()
        engine = _engine()

        query_vec = embed_text(engine.model, engine.tokenizer, req.query, device=device)
        results = []
        for i, doc in enumerate(req.documents):
            doc_vec = embed_text(engine.model, engine.tokenizer, doc, device=device)
            results.append(RerankResult(index=i, relevance_score=cosine_similarity(query_vec, doc_vec)))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        if req.top_n is not None:
            results = results[: req.top_n]

        return RerankResponse(model=req.model, results=results)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        REQUESTS_TOTAL.labels(endpoint="chat_completions").inc()
        engine = _engine()
        tokenizer = engine.tokenizer

        template_kwargs = {"tools": req.tools} if req.tools else {}
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in req.messages],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )["input_ids"][0].tolist()

        lora_id = _resolve_lora_id(req.model, state, engine)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        start_time = time.monotonic()

        if req.use_beam_search:
            _reject_unsupported_beam_search_combo(req)
            text, finish_reason = await engine.generate_beam_search(
                prompt_ids,
                max_new_tokens=req.max_tokens,
                num_beams=req.best_of or 4,
                length_penalty=req.length_penalty,
                eos_token_id=tokenizer.eos_token_id,
                lora_id=lora_id,
            )
            completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            TOKENS_GENERATED_TOTAL.inc(completion_tokens)
            REQUEST_LATENCY_SECONDS.labels(endpoint="chat_completions").observe(time.monotonic() - start_time)
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=text), finish_reason=finish_reason)],
                usage=UsageInfo(
                    prompt_tokens=len(prompt_ids), completion_tokens=completion_tokens,
                    total_tokens=len(prompt_ids) + completion_tokens,
                ),
            )

        stop_strings = _normalize_stop(req.stop)
        # Once a tool call's closing tag appears there's nothing left worth
        # generating -- stop there (inclusively: the tag itself has to
        # survive for parse_tool_calls to find it, see stop_strings.py).
        inclusive_stop_strings = ["</tool_call>"] if req.tools else []
        sampling_params = _sampling_params_from(req)
        guide = _build_guide(req, engine, tokenizer_id or model_id)

        raw_gen = engine.generate(
            prompt_ids,
            max_new_tokens=req.max_tokens,
            sampling_params=sampling_params,
            eos_token_id=tokenizer.eos_token_id,
            guide=guide,
            lora_id=lora_id,
        )
        filtered = StopStringFilter(raw_gen, stop_strings, inclusive_stop_strings=inclusive_stop_strings)

        if req.stream:

            async def event_stream():
                first_token = True
                full_text_parts = []
                async for delta in filtered:
                    if first_token:
                        TIME_TO_FIRST_TOKEN_SECONDS.observe(time.monotonic() - start_time)
                        first_token = False
                    full_text_parts.append(delta)
                    TOKENS_GENERATED_TOTAL.inc()
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=req.model,
                        choices=[ChatCompletionChunkChoice(delta=DeltaMessage(content=delta))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # No incremental tool_calls deltas (a real OpenAI streaming
                # client gets those piece by piece) -- deferred; the full
                # parsed tool call is attached to this one final chunk
                # instead, which is enough for a client that waits for the
                # complete response before acting on it.
                final_delta = DeltaMessage()
                finish_reason = "stop" if filtered.stopped_early else "length"
                if filtered.stopped_inclusive:
                    _, tool_calls = parse_tool_calls("".join(full_text_parts))
                    if tool_calls:
                        final_delta = DeltaMessage(tool_calls=tool_calls)
                        finish_reason = "tool_calls"
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[ChatCompletionChunkChoice(delta=final_delta, finish_reason=finish_reason)],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                REQUEST_LATENCY_SECONDS.labels(endpoint="chat_completions").observe(time.monotonic() - start_time)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        chunks = []
        first_token = True
        async for delta in filtered:
            if first_token:
                TIME_TO_FIRST_TOKEN_SECONDS.observe(time.monotonic() - start_time)
                first_token = False
            chunks.append(delta)
        text = "".join(chunks)
        completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        TOKENS_GENERATED_TOTAL.inc(completion_tokens)
        REQUEST_LATENCY_SECONDS.labels(endpoint="chat_completions").observe(time.monotonic() - start_time)

        content, tool_calls = (text, None)
        finish_reason = "stop" if filtered.stopped_early else "length"
        if filtered.stopped_inclusive:
            content, tool_calls = parse_tool_calls(text)
            if tool_calls:
                finish_reason = "tool_calls"
            else:
                content = text  # malformed block after all -- fall back to raw text

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt_ids) + completion_tokens,
            ),
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        REQUESTS_TOTAL.labels(endpoint="completions").inc()
        engine = _engine()
        tokenizer = engine.tokenizer

        prompt_ids = tokenizer.encode(req.prompt, add_special_tokens=False)
        lora_id = _resolve_lora_id(req.model, state, engine)
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        start_time = time.monotonic()

        if req.use_beam_search:
            _reject_unsupported_beam_search_combo(req)
            text, finish_reason = await engine.generate_beam_search(
                prompt_ids,
                max_new_tokens=req.max_tokens,
                num_beams=req.best_of or 4,
                length_penalty=req.length_penalty,
                eos_token_id=tokenizer.eos_token_id,
                lora_id=lora_id,
            )
            completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            TOKENS_GENERATED_TOTAL.inc(completion_tokens)
            REQUEST_LATENCY_SECONDS.labels(endpoint="completions").observe(time.monotonic() - start_time)
            return CompletionResponse(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[CompletionChoice(text=text, finish_reason=finish_reason)],
                usage=UsageInfo(
                    prompt_tokens=len(prompt_ids), completion_tokens=completion_tokens,
                    total_tokens=len(prompt_ids) + completion_tokens,
                ),
            )

        stop_strings = _normalize_stop(req.stop)
        sampling_params = _sampling_params_from(req)
        guide = _build_guide(req, engine, tokenizer_id or model_id)

        raw_gen = engine.generate(
            prompt_ids,
            max_new_tokens=req.max_tokens,
            sampling_params=sampling_params,
            eos_token_id=tokenizer.eos_token_id,
            guide=guide,
            lora_id=lora_id,
        )
        filtered = StopStringFilter(raw_gen, stop_strings)

        if req.stream:

            async def event_stream():
                async for delta in filtered:
                    TOKENS_GENERATED_TOTAL.inc()
                    chunk = CompletionChunk(
                        id=completion_id,
                        created=created,
                        model=req.model,
                        choices=[CompletionChunkChoice(text=delta)],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                finish_reason = "stop" if filtered.stopped_early else "length"
                final_chunk = CompletionChunk(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[CompletionChunkChoice(text="", finish_reason=finish_reason)],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                REQUEST_LATENCY_SECONDS.labels(endpoint="completions").observe(time.monotonic() - start_time)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        chunks = [delta async for delta in filtered]
        text = "".join(chunks)
        completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        TOKENS_GENERATED_TOTAL.inc(completion_tokens)
        REQUEST_LATENCY_SECONDS.labels(endpoint="completions").observe(time.monotonic() - start_time)

        finish_reason = "stop" if filtered.stopped_early else "length"
        return CompletionResponse(
            id=completion_id,
            created=created,
            model=req.model,
            choices=[CompletionChoice(text=text, finish_reason=finish_reason)],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt_ids) + completion_tokens,
            ),
        )

    return app
