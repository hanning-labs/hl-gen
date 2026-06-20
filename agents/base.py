"""Agent role hierarchy.

Maps to Table B.1 of the framework: a base :class:`Agent` plus one class per
agent *type* — Generator, Scorer, Reducer, Editor, Sink. Concrete agents
subclass these role bases; only the single abstract async method on each needs
implementing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from llm.base import LLMClient, Message
from models import (
    AgentScore,
    CSSample,
    GenerationContext,
    RefinementFeedback,
    ScoreReport,
)
from prompting import PromptParseError, parse_json

log = logging.getLogger(__name__)


class Agent(ABC):
    """Common base for every agent in the pipeline."""

    #: Human-readable agent name (e.g. "FluencyAgent"); overridden per agent.
    name: str = "Agent"

    #: How many times to retry an LLM call when the reply fails to parse.
    parse_retries: int = 2

    def __init__(self, llm: LLMClient, *, name: str | None = None, parse_retries: int | None = None) -> None:
        self.llm = llm
        if name is not None:
            self.name = name
        if parse_retries is not None:
            self.parse_retries = parse_retries

    async def _complete_with_retry(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        validate: Callable[[Any], Any] = lambda x: x,
        **kwargs: Any,
    ) -> Any:
        """Call LLM, parse JSON reply, run ``validate``; retry up to ``parse_retries`` times.

        ``validate`` receives the parsed value and should raise ``PromptParseError``
        if the shape is wrong. Its return value is forwarded to the caller.
        """
        last_exc: PromptParseError | None = None
        for attempt in range(self.parse_retries + 1):
            response = await self.llm.complete(messages, system=system, **kwargs)
            try:
                result = parse_json(response.text)
                return validate(result)
            except PromptParseError as exc:
                last_exc = exc
                log.warning(
                    "%s parse failure (attempt %d/%d): %s",
                    self.name, attempt + 1, self.parse_retries + 1, exc,
                )
        assert last_exc is not None
        raise last_exc


class GeneratorAgent(Agent):
    """Produces a CS sample from topic/persona/languages (and feedback)."""

    @abstractmethod
    async def generate(self, ctx: GenerationContext) -> CSSample:
        ...


class ScorerAgent(Agent):
    """Scores a single quality dimension of a sample."""

    #: Relative weight in the SummarizeAgent's weighted mean.
    weight: float = 1.0

    def __init__(
        self, llm: LLMClient, *, name: str | None = None, weight: float | None = None, parse_retries: int | None = None
    ) -> None:
        super().__init__(llm, name=name, parse_retries=parse_retries)
        if weight is not None:
            self.weight = weight

    @abstractmethod
    async def score(self, sample: CSSample) -> AgentScore:
        ...


class ReducerAgent(Agent):
    """Normalizes per-scorer scores into a single ScoreReport (S_final)."""

    @abstractmethod
    async def summarize(self, scores: list[AgentScore], *, threshold: float) -> ScoreReport:
        ...


class EditorAgent(Agent):
    """Turns a failing report into actionable refinement feedback."""

    @abstractmethod
    async def refine(self, sample: CSSample, report: ScoreReport) -> RefinementFeedback:
        ...


class SinkAgent(Agent):
    """Terminal agent: persists accepted samples and logs metadata."""

    @abstractmethod
    async def accept(self, sample: CSSample, report: ScoreReport, tool_context: dict) -> None:
        ...
