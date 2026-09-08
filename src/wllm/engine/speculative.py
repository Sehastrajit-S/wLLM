"""N-gram (prompt-lookup) speculative decoding: no draft model to load or
orchestrate -- search the sequence's own token history for a previous
occurrence of its current tail, and propose whatever followed that
occurrence last time as a draft continuation. Works well for repetitive
generation (code edits, structured output, anything echoing its input) and
degrades to "no draft, decode normally" whenever nothing matches.

Verification reuses continue_prefill (the same "attend to cached context +
causally within the new chunk" primitive chunked prefill and prefix caching
already use) -- feeding [last_real_token, *draft_tokens] gets back one
prediction per position, compared against the actually-drafted next token at
each step. See Scheduler._speculative_decode_one for how the accept/reject
math and cache rollback fit together.
"""
from __future__ import annotations


def propose_ngram_draft(history: list[int], n: int, k: int) -> list[int] | None:
    """`history` is the sequence's full token list so far (prompt +
    generated). Looks for the most recent earlier occurrence of the last `n`
    tokens elsewhere in `history`, and if found, returns up to `k` tokens
    that followed it. None if history is too short or nothing matches.
    """
    if len(history) < n + 1:
        return None

    query = tuple(history[-n:])
    search_space = history[:-n]

    for start in range(len(search_space) - n, -1, -1):
        if tuple(search_space[start : start + n]) == query:
            candidate_start = start + n
            draft = history[candidate_start : candidate_start + k]
            if draft:
                return draft

    return None
