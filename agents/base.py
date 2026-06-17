"""Agent role hierarchy.

Maps to Table B.1 of the framework: a base :class:`Agent` plus one class per
agent *type* — Generator, Scorer, Reducer, Editor, Sink. Concrete agents
subclass these role bases; only the single abstract async method on each needs
implementing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..llm.base import LLMClient
from ..models import (
    AgentScore,
    CSSample,
    GenerationContext,
    RefinementFeedback,
    ScoreReport,
)


class Agent(ABC):
    """Common base for every agent in the pipeline."""

    #: Human-readable agent name (e.g. "FluencyAgent"); overridden per agent.
    name: str = "Agent"

    def __init__(self, llm: LLMClient, *, name: str | None = None) -> None:
        self.llm = llm
        if name is not None:
            self.name = name


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
        self, llm: LLMClient, *, name: str | None = None, weight: float | None = None
    ) -> None:
        super().__init__(llm, name=name)
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
    async def accept(self, sample: CSSample, report: ScoreReport) -> None:
        ...
