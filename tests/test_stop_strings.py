import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from wllm.api.stop_strings import StopStringFilter


async def _chunks(*items):
    for item in items:
        yield item


async def _collect(filtered):
    return [chunk async for chunk in filtered]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_exclusive_stop_string_cuts_before_the_match():
    gen = _chunks("The ", "capital ", "of ", "France ", "is ", "Paris", ".", " More text")
    filtered = StopStringFilter(gen, stop_strings=["."])
    result = "".join(await _collect(filtered))
    assert result == "The capital of France is Paris"
    assert filtered.stopped_early
    assert not filtered.stopped_inclusive


@pytest.mark.anyio
async def test_inclusive_stop_string_keeps_the_match_and_cuts_after_it():
    gen = _chunks("<tool_call>", '{"name": "f", "arguments": {}}', "</tool_call>", "\nHuman: garbage")
    filtered = StopStringFilter(gen, stop_strings=[], inclusive_stop_strings=["</tool_call>"])
    result = "".join(await _collect(filtered))
    assert result == '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    assert filtered.stopped_early
    assert filtered.stopped_inclusive


@pytest.mark.anyio
async def test_inclusive_match_spanning_a_chunk_boundary():
    gen = _chunks("<tool_call>{}</tool", "_call>", "trailing junk")
    filtered = StopStringFilter(gen, stop_strings=[], inclusive_stop_strings=["</tool_call>"])
    result = "".join(await _collect(filtered))
    assert result == "<tool_call>{}</tool_call>"


@pytest.mark.anyio
async def test_no_match_passes_everything_through():
    gen = _chunks("no ", "stop ", "here")
    filtered = StopStringFilter(gen, stop_strings=["</tool_call>"], inclusive_stop_strings=["ZZZ"])
    result = "".join(await _collect(filtered))
    assert result == "no stop here"
    assert not filtered.stopped_early
