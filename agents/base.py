"""Agent role hierarchy.

Maps to Table B.1 of the framework: a base :class:`Agent` plus one class per
agent *type* — Generator, Scorer, Reducer, Editor, Sink. Concrete agents
subclass these role bases; only the single abstract async method on each needs
implementing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from llm.base import LLMClient, Message
from models import (
    AgentScore,
    CSSample,
    GenerationContext,
    RefinementFeedback,
    ScoreReport,
)
from prompting import PromptParseError, as_user, parse_json

log = logging.getLogger(__name__)

_GUIDES_DIR = Path(__file__).resolve().parent / "guides"


def load_guide(filename: str) -> str:
    """Read a prompt/guide template from agents/guides/."""
    return (_GUIDES_DIR / filename).read_text()


def fill_template(template: str, **values: object) -> str:
    """Substitute ``{key}`` placeholders by literal replacement, leaving other braces untouched."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


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
                self._last_llm_response = response
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


_SCORER_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences. Use true/false for all criterion fields."
_REFINER_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."


class DimensionScorer(ScorerAgent):
    """Shared scoring skeleton for boolean-criteria judges.

    Subclasses set ``prompt``, ``criteria``, and implement ``_format_prompt``.
    """

    prompt: str = ""
    criteria: tuple[str, ...] = ()

    #: System prompt for the judging call; subclasses may override to add guidance.
    system: str = _SCORER_SYSTEM

    @abstractmethod
    def _format_prompt(self, sample: CSSample) -> str: ...

    async def score(self, sample: CSSample) -> AgentScore:
        criteria = self.criteria
        name = self.name

        def _validate(data: object) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"{name}: expected a JSON object, got {type(data).__name__}")
            for k in criteria:
                if k not in data:
                    raise PromptParseError(f"{name}: missing criterion {k!r} in model reply")
            return data

        data = await self._complete_with_retry(as_user(self._format_prompt(sample)), system=self.system, validate=_validate)
        passed = sum(bool(data[k]) for k in self.criteria)
        score = passed / len(self.criteria) * 10
        log.debug("%s score=%.1f (%d/%d)", self.name, score, passed, len(self.criteria))
        return AgentScore(agent=self.name, score=score, rationale=data.get("notes", ""))


class RefinerBase(EditorAgent):
    """Shared refiner skeleton.

    Subclasses implement ``_fill_prompt``; ``_summarize_scores`` and ``refine`` are provided.
    """

    @staticmethod
    def _summarize_scores(report: ScoreReport) -> str:
        lines = [
            f"- {s.agent}: {s.score:.1f}/10 — {s.rationale or '(no notes)'}"
            for s in sorted(report.scores, key=lambda s: s.score)
        ]
        return "\n".join(lines) if lines else "(no scorer feedback available)"

    @abstractmethod
    def _fill_prompt(self, sample: CSSample, report: ScoreReport) -> str: ...

    async def refine(self, sample: CSSample, report: ScoreReport) -> RefinementFeedback:
        name = self.name

        def _validate(data: object) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"{name}: expected a JSON object, got {type(data).__name__}")
            return data

        data = await self._complete_with_retry(as_user(self._fill_prompt(sample, report)), system=_REFINER_SYSTEM, validate=_validate)

        failures = data.get("failures") or []
        if isinstance(failures, str):
            failures = [failures]
        elif not isinstance(failures, list):
            failures = [str(failures)]
        failures = [str(f) for f in failures if f]

        suggestions = data.get("suggestions") or ""
        if not isinstance(suggestions, str):
            suggestions = str(suggestions)

        return RefinementFeedback(failures=failures, suggestions=suggestions)
