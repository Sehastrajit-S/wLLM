"""OpenAI's `stop` parameter is text-based (unlike our engine's token-id-based
`stop_token_ids`), and a stop string can span a chunk boundary or not align
with any single token. This wraps a raw text-delta generator, buffers enough
to detect a stop string across chunk boundaries, and truncates output at the
first match.

`inclusive_stop_strings` are a second, independent set that stop generation
the same way but keep the matched text itself in the output, cut right after
it ends rather than right before it starts. Tool calling needs this: once a
`</tool_call>` closing tag appears there's nothing more worth generating, but
the tag itself has to survive in the text or the response parser has nothing
to match against.

Explicitly closes the wrapped generator (rather than relying on GC) once a
stop string is found or the caller stops consuming early -- `break`ing an
`async for` does NOT call `aclose()` on the generator being iterated, it
just abandons it suspended; without an explicit close, the underlying
sequence would keep running in the scheduler and holding cache blocks
until it separately finishes on its own. Confirmed the hard way while
testing AsyncEngine.generate's own cancel-on-early-exit path.
"""
from __future__ import annotations


class StopStringFilter:
    def __init__(self, gen, stop_strings: list[str], inclusive_stop_strings: list[str] | None = None):
        self._gen = gen
        self._stop_strings = [s for s in stop_strings if s]
        self._inclusive_stop_strings = [s for s in (inclusive_stop_strings or []) if s]
        self.stopped_early = False
        self.stopped_inclusive = False

    async def __aiter__(self):
        buffer = ""
        closed = False
        try:
            async for delta in self._gen:
                buffer += delta
                visible_so_far = len(buffer) - len(delta)

                inclusive_cut = None
                for s in self._inclusive_stop_strings:
                    idx = buffer.find(s)
                    if idx != -1:
                        end = idx + len(s)
                        inclusive_cut = end if inclusive_cut is None else min(inclusive_cut, end)

                if inclusive_cut is not None:
                    if inclusive_cut > visible_so_far:
                        yield buffer[visible_so_far:inclusive_cut]
                    self.stopped_early = True
                    self.stopped_inclusive = True
                    closed = True
                    await self._gen.aclose()
                    return

                cut = None
                for s in self._stop_strings:
                    idx = buffer.find(s)
                    if idx != -1:
                        cut = idx if cut is None else min(cut, idx)
                if cut is not None:
                    if cut > visible_so_far:
                        yield buffer[visible_so_far:cut]
                    self.stopped_early = True
                    closed = True
                    await self._gen.aclose()
                    return
                yield delta
        finally:
            if not closed:
                await self._gen.aclose()
