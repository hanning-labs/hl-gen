"""Scorer agents for the English topics pipeline.

Four dimensions: topic relevance, coherence, factual quality, style adherence.
All inherit _TopicDimensionScorer which omits the CS-specific cs_ratio placeholder.
"""

from __future__ import annotations

import logging

from models import AgentScore, CSSample
from prompting import PromptParseError, as_user, json_only_instruction
from agents.base import ScorerAgent

log = logging.getLogger(__name__)

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."


class _TopicDimensionScorer(ScorerAgent):
    """Shared scoring path for a single-dimension topic judge."""

    prompt: str = ""
    score_key: str = "score"
    rationale_keys: tuple[str, ...] = ()
    response_shape: str = ""

    def _format_prompt(self, sample: CSSample) -> str:
        body = self.prompt.format(
            text=sample.text,
            topic=sample.request.topic,
            style=sample.request.style,
        )
        return f"{body}\n\n{json_only_instruction(self.response_shape)}"

    def _rationale(self, data: dict) -> str:
        parts = []
        for key in self.rationale_keys:
            value = data.get(key)
            if value:
                parts.append(value if isinstance(value, str) else str(value))
        return " — ".join(parts)

    async def score(self, sample: CSSample) -> AgentScore:
        score_key = self.score_key
        name = self.name

        def _validate(data: object) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"{name}: expected a JSON object, got {type(data).__name__}")
            if score_key not in data:
                raise PromptParseError(f"{name}: missing {score_key!r} in model reply")
            try:
                float(data[score_key])
            except (TypeError, ValueError) as exc:
                raise PromptParseError(f"{name}: non-numeric {score_key!r} in model reply") from exc
            return data

        data = await self._complete_with_retry(as_user(self._format_prompt(sample)), system=_SYSTEM, validate=_validate)
        score = max(0.0, min(10.0, float(data[self.score_key])))
        log.debug("%s score=%.1f for topic=%r", self.name, score, sample.request.topic)
        return AgentScore(agent=self.name, score=score, rationale=self._rationale(data))


RELEVANCE_PROMPT = (
    "You are **TopicRelevanceAgent**. Evaluate whether the following text actually addresses the stated topic.\n\n"
    "Topic: {topic}\n"
    "Text: {text}\n\n"
    "1. Does the text directly engage with the topic?\n"
    "2. Are the key concepts of the topic present?\n"
    "3. Does the text stay on-topic throughout?\n\n"
    "Output:\n"
    "- A `relevance_score` (0 to 10).\n"
    "- A list of `issues` (if any) where the text drifts off-topic.\n"
    "- A short `summary` of overall topic alignment."
)

COHERENCE_PROMPT = (
    "You are **CoherenceAgent**. Evaluate the logical structure and clarity of the following text.\n\n"
    "Text: {text}\n\n"
    "1. Does the text flow logically from one idea to the next?\n"
    "2. Are transitions clear and the argument easy to follow?\n"
    "3. Is the text free of contradictions or unclear references?\n\n"
    "Output:\n"
    "- A `coherence_score` (0 to 10).\n"
    "- A list of `issues` (if any) with specific structural problems.\n"
    "- A short `summary` of overall coherence."
)

FACTUAL_PROMPT = (
    "You are **FactualQualityAgent**. Evaluate whether the claims in the following text are grounded and credible.\n\n"
    "Text: {text}\n\n"
    "1. Are the factual claims plausible and specific (not vague filler)?\n"
    "2. Are there any obvious hallucinations, invented statistics, or unsupported assertions?\n"
    "3. Does the text cite or reference real context where appropriate?\n\n"
    "Output:\n"
    "- A `factual_score` (0 to 10).\n"
    "- A list of `issues` (if any) naming specific questionable claims.\n"
    "- A short `summary` of overall factual quality."
)

STYLE_PROMPT = (
    "You are **StyleAdherenceAgent**. Evaluate whether the following text matches the requested style.\n\n"
    "Requested style: {style}\n"
    "Text: {text}\n\n"
    "Ask and answer the following:\n"
    "1. Does the text conform to the structure of the requested style?\n"
    "2. Is the tone and voice appropriate for the style?\n\n"
    "Output:\n"
    "- A `style_score` (0 to 10).\n"
    "- A list of `issues` (if any) naming specific style mismatches.\n"
    "- A short `summary` of overall style fit."
)


class TopicRelevanceAgent(_TopicDimensionScorer):
    name = "TopicRelevanceAgent"
    prompt = RELEVANCE_PROMPT
    score_key = "relevance_score"
    rationale_keys = ("summary", "issues")
    response_shape = '{"relevance_score": <0-10>, "issues": ["..."], "summary": "..."}'


class CoherenceAgent(_TopicDimensionScorer):
    name = "CoherenceAgent"
    prompt = COHERENCE_PROMPT
    score_key = "coherence_score"
    rationale_keys = ("summary", "issues")
    response_shape = '{"coherence_score": <0-10>, "issues": ["..."], "summary": "..."}'


class FactualQualityAgent(_TopicDimensionScorer):
    name = "FactualQualityAgent"
    prompt = FACTUAL_PROMPT
    score_key = "factual_score"
    rationale_keys = ("summary", "issues")
    response_shape = '{"factual_score": <0-10>, "issues": ["..."], "summary": "..."}'


class StyleAdherenceAgent(_TopicDimensionScorer):
    name = "StyleAdherenceAgent"
    prompt = STYLE_PROMPT
    score_key = "style_score"
    rationale_keys = ("summary", "issues")
    response_shape = '{"style_score": <0-10>, "issues": ["..."], "summary": "..."}'
