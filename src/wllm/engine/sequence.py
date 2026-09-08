from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from wllm.engine.sampling import SamplingParams

if TYPE_CHECKING:
    from wllm.engine.guided_decoding import JSONSchemaGuide


class SeqStatus(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"  # admitted, mid-way through chunked prefill -- not yet decodable
    RUNNING = "running"
    SWAPPED = "swapped"  # preempted, KV cache saved to host RAM -- progress preserved, awaiting swap-in
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None
    stop_token_ids: frozenset[int] = frozenset()
    sampling_params: SamplingParams = field(default_factory=SamplingParams)  # defaults to greedy
    output_token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    guide: JSONSchemaGuide | None = None
    lora_id: str | None = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_len_estimate(self) -> int:
        """Conservative upper bound used for admission control -- this is the
        max number of tokens this sequence could ever occupy in the cache.
        """
        return self.num_prompt_tokens + self.max_new_tokens

    @property
    def last_token_id(self) -> int:
        return (self.output_token_ids or self.prompt_token_ids)[-1]

    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED

    def append_token(self, token_id: int, logprob: float = 0.0) -> None:
        self.output_token_ids.append(token_id)
        self.logprobs.append(logprob)
        if self.guide is not None:
            self.guide.advance(token_id)
        if self.eos_token_id is not None and token_id == self.eos_token_id or token_id in self.stop_token_ids or self.guide is not None and self.guide.is_finished() or len(self.output_token_ids) >= self.max_new_tokens:
            self.status = SeqStatus.FINISHED
