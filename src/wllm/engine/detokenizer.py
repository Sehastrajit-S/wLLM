"""Incremental (streaming) detokenization. Decoding tokens one at a time in
isolation can produce garbled text -- BPE merges and spacing depend on
neighboring tokens, and a single new token can retroactively change how the
previous one should render. The standard fix (same approach HF's
TextIteratorStreamer and vLLM use): re-decode the *entire* sequence so far
every step and emit only the text suffix that's new versus last time. O(n)
work per step, O(n^2) total for a full generation -- correctness-first,
not yet optimized.
"""
from __future__ import annotations


class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.prev_text: str = ""

    def add_token(self, token_id: int) -> str:
        """Appends `token_id` and returns the new text produced by it (which
        may be an empty string, if the token doesn't complete a renderable
        unit yet -- e.g. one half of a multi-byte character).
        """
        self.token_ids.append(token_id)
        full_text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
        delta = full_text[len(self.prev_text) :]
        self.prev_text = full_text
        return delta
