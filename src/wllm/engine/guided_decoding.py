"""JSON-schema-constrained decoding. Schema/grammar compilation (JSON schema
-> regex -> a per-token-step allowed-token-set automaton) is genuinely
complex machinery (essentially a regex-to-FSM compiler) with no real
engineering value in reimplementing -- `outlines_core` (a small, focused,
Rust-backed library) does exactly this and nothing else. We use it purely
for that compilation step; the actual integration into our sampler and
scheduler (masking logits, per-sequence guide state, stopping when the
guide reports completion) is ours.

Only JSON-schema (via a compiled regex) is supported -- not arbitrary
context-free grammars (e.g. GBNF). That covers the overwhelming majority of
real "structured output" requests; full CFG support is a larger, separate
undertaking and is deferred, same as beam search and multi-LoRA before it.
"""
from __future__ import annotations

import outlines_core
import torch


class JSONSchemaGuide:
    """One instance per constrained sequence -- the automaton's state is
    inherently per-sequence (different sequences are at different points in
    their JSON output), so this can't be shared like the tokenizer/vocab can.
    """

    _vocab_cache: dict[str, outlines_core.Vocabulary] = {}

    def __init__(self, tokenizer_name_or_path: str, json_schema: str | dict, vocab_size: int):
        import json as json_module

        if isinstance(json_schema, dict):
            json_schema = json_module.dumps(json_schema)

        vocab = self._get_vocab(tokenizer_name_or_path)
        regex = outlines_core.json_schema.build_regex_from_schema(json_schema)
        index = outlines_core.Index(regex, vocab)
        self._guide = outlines_core.Guide(index)
        self.vocab_size = vocab_size

    @classmethod
    def _get_vocab(cls, name: str):
        if name not in cls._vocab_cache:
            cls._vocab_cache[name] = outlines_core.Vocabulary.from_pretrained(name)
        return cls._vocab_cache[name]

    def allowed_token_mask(self, device: str = "cuda") -> torch.Tensor:
        """(vocab_size,) bool tensor, True where the token is a valid next
        token under the schema's current state.
        """
        allowed = self._guide.get_tokens()
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        mask[torch.tensor(allowed, dtype=torch.long, device=device)] = True
        return mask

    def advance(self, token_id: int) -> None:
        self._guide.advance(token_id)

    def is_finished(self) -> bool:
        return self._guide.is_finished()
