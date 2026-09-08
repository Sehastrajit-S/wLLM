import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wllm.api.tool_calls import parse_tool_calls


def test_no_tool_call_returns_text_unchanged():
    text = "The capital of France is Paris."
    content, tool_calls = parse_tool_calls(text)
    assert content == text
    assert tool_calls is None


def test_single_tool_call_is_parsed():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
    content, tool_calls = parse_tool_calls(text)
    assert content is None
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call.type == "function"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"location": "Paris"}
    assert call.id.startswith("call_")


def test_leading_text_before_tool_call_is_preserved():
    text = 'Let me check that for you.\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
    content, tool_calls = parse_tool_calls(text)
    assert content == "Let me check that for you."
    assert len(tool_calls) == 1


def test_multiple_tool_calls_are_all_parsed():
    text = (
        '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {"y": 2}}\n</tool_call>'
    )
    content, tool_calls = parse_tool_calls(text)
    assert content is None
    assert [c.function.name for c in tool_calls] == ["a", "b"]
    assert json.loads(tool_calls[0].function.arguments) == {"x": 1}
    assert json.loads(tool_calls[1].function.arguments) == {"y": 2}


def test_tool_call_ids_are_unique():
    text = (
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>'
    )
    _, tool_calls = parse_tool_calls(text)
    assert tool_calls[0].id != tool_calls[1].id


def test_malformed_json_block_is_skipped_not_fatal():
    text = '<tool_call>\nnot valid json\n</tool_call>'
    content, tool_calls = parse_tool_calls(text)
    # no valid tool call extracted -- falls back to returning the raw text
    assert tool_calls is None
    assert content == text


def test_missing_name_field_is_skipped():
    text = '<tool_call>\n{"arguments": {}}\n</tool_call>'
    content, tool_calls = parse_tool_calls(text)
    assert tool_calls is None


def test_arguments_default_to_empty_object_when_absent():
    text = '<tool_call>\n{"name": "ping"}\n</tool_call>'
    _, tool_calls = parse_tool_calls(text)
    assert json.loads(tool_calls[0].function.arguments) == {}
