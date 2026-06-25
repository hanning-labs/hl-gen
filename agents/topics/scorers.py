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

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences. Use true/false for all criterion fields."


class _TopicDimensionScorer(ScorerAgent):
    """Shared scoring path for a single-dimension topic judge."""

    prompt: str = ""
    criteria: tuple[str, ...] = ()

    def _format_prompt(self, sample: CSSample) -> str:
        body = self.prompt.format(
            text=sample.text,
            topic=sample.request.topic,
            style=sample.request.style,
        )
        shape = "{" + ", ".join(f'"{k}": true/false' for k in self.criteria) + ', "notes": "..."}'
        return f"{body}\n\n{json_only_instruction(shape)}"

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

        data = await self._complete_with_retry(as_user(self._format_prompt(sample)), system=_SYSTEM, validate=_validate)
        passed = sum(bool(data[k]) for k in self.criteria)
        score = passed / len(self.criteria) * 10
        log.debug("%s score=%.1f (%d/%d) for topic=%r", self.name, score, passed, len(self.criteria), sample.request.topic)
        return AgentScore(agent=self.name, score=score, rationale=data.get("notes", ""))


RELEVANCE_PROMPT = (
    "You are **TopicRelevanceAgent**. Evaluate whether the following text actually addresses the stated topic.\n\n"
    "Topic: {topic}\n"
    "Text: {text}\n\n"
    "For each criterion below, answer true or false. Add a \"notes\" field with 1–2 sentences.\n"
    "- addresses_topic: The text directly engages with the stated topic\n"
    "- key_concepts_present: Key concepts of the topic are present\n"
    "- stays_on_topic: The text stays on-topic throughout"
)

COHERENCE_PROMPT = (
    "You are **CoherenceAgent**. Evaluate the logical structure and clarity of the following text.\n\n"
    "Text: {text}\n\n"
    "For each criterion below, answer true or false. Add a \"notes\" field with 1–2 sentences.\n"
    "- logical_flow: Ideas flow logically from one to the next\n"
    "- clear_transitions: Transitions are clear and the argument is easy to follow\n"
    "- no_contradictions: Text is free of contradictions or unclear references"
)

DEPTH_PROMPT = (
    "You are **DepthAgent**. Evaluate whether the following text develops its ideas "
    "or stays at a surface level.\n\n"
    "Text: {text}\n\n"
    "For each criterion below, answer true or false. Add a \"notes\" field with 1–2 sentences.\n"
    "- develops_ideas: Ideas are explored and developed, not just named or stated\n"
    "- specific_details: Content includes specific details, examples, or supporting points\n"
    "- avoids_obvious: Content goes beyond restating the obvious or the headline"
)

STYLE_PROMPT = (
    "You are **StyleAdherenceAgent**. Evaluate whether the following text matches the requested style.\n\n"
    "Requested style: {style}\n"
    "Text: {text}\n\n"
    "For each criterion below, answer true or false. Add a \"notes\" field with 1–2 sentences.\n"
    "- structure_conforms: Text conforms to the structure of the requested style\n"
    "- tone_appropriate: Tone and voice are appropriate for the style"
)


class TopicRelevanceAgent(_TopicDimensionScorer):
    name = "TopicRelevanceAgent"
    prompt = RELEVANCE_PROMPT
    criteria = ("addresses_topic", "key_concepts_present", "stays_on_topic")


class CoherenceAgent(_TopicDimensionScorer):
    name = "CoherenceAgent"
    prompt = COHERENCE_PROMPT
    criteria = ("logical_flow", "clear_transitions", "no_contradictions")


class DepthAgent(_TopicDimensionScorer):
    name = "DepthAgent"
    prompt = DEPTH_PROMPT
    criteria = ("develops_ideas", "specific_details", "avoids_obvious")


class StyleAdherenceAgent(_TopicDimensionScorer):
    name = "StyleAdherenceAgent"
    prompt = STYLE_PROMPT
    criteria = ("structure_conforms", "tone_appropriate")
