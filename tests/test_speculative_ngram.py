import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wllm.engine.speculative import propose_ngram_draft


def test_history_too_short_returns_none():
    assert propose_ngram_draft([1, 2], n=3, k=2) is None


def test_no_repeated_ngram_returns_none():
    assert propose_ngram_draft([1, 2, 3, 4, 5, 6, 7], n=3, k=2) is None


def test_clear_repeat_proposes_what_followed_before():
    history = [1, 2, 3, 4, 5, 1, 2, 3]
    draft = propose_ngram_draft(history, n=3, k=2)
    assert draft == [4, 5]


def test_draft_truncated_to_available_history_after_match():
    # earlier match of "1,2" is at the very start, so only 3 tokens ([9,1,2])
    # exist after it even though k=5 is requested -- must not index past the end
    history = [1, 2, 9, 1, 2]
    draft = propose_ngram_draft(history, n=2, k=5)
    assert draft == [9, 1, 2]


def test_most_recent_match_preferred_over_older_one():
    # "1,2,3" appears twice earlier: followed by [4,5] the first time, [9,9] the second (most recent)
    history = [1, 2, 3, 4, 5, 1, 2, 3, 9, 9, 1, 2, 3]
    draft = propose_ngram_draft(history, n=3, k=2)
    assert draft == [9, 9]


def test_match_immediately_before_current_tail_is_found():
    history = [5, 5, 5, 1, 2, 3, 1, 2, 3]
    draft = propose_ngram_draft(history, n=3, k=1)
    assert draft == [1]
