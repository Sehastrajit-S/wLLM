"""Parses Qwen2.5's native tool-call output format into OpenAI's
`tool_calls` response structure. The prompt side needs nothing custom --
Qwen's own chat template already knows how to render a `tools` list (passed
straight through to `apply_chat_template`) and instructs the model to
respond with:

    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>

one such block per call. This module only handles the output side: finding
those blocks and converting them to `[{"id", "type", "function": {"name",
"arguments"}}]`, where `arguments` is a JSON *string* per the OpenAI spec
(not a nested object, unlike Qwen's own wire format above).

Only this one model family's native format is handled -- other models use
different tags (or none, relying purely on guided-decoding-style
enforcement). Broader multi-format support is a straightforward extension
if/when other model families are added, not implemented speculatively here.
"""
from __future__ import annotations

import json
import re
import uuid

from wllm.api.schemas import FunctionCall, ToolCall

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str | None, list[ToolCall] | None]:
    """Returns (remaining_content, tool_calls). remaining_content is the text
    before the first tool-call block (or None if empty/absent); tool_calls is
    None if no valid tool-call block was found anywhere in `text`.
    """
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return text, None

    tool_calls = []
    for m in matches:
        try:
            payload = json.loads(m.group(1))
            name = payload["name"]
            arguments = payload.get("arguments", {})
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # malformed block from the model -- skip rather than fail the whole response
        tool_calls.append(
            ToolCall(id=f"call_{uuid.uuid4().hex}", function=FunctionCall(name=name, arguments=json.dumps(arguments)))
        )

    if not tool_calls:
        return text, None

    leading_text = text[: matches[0].start()].strip()
    return (leading_text or None), tool_calls
