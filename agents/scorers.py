"""Scorer agents (Table B.1): Fluency, Naturalness, CSRatio, SocialCulture.

Each scores one quality dimension on a 0–10 scale; the SummarizeAgent combines
them into S_final. All four share the same shape — show the model the spec, the
utterance, and a dimension-specific rubric, and parse a ``{score, rationale}``
reply — so the common path lives in :class:`_DimensionScorer` and each concrete
agent only supplies its ``name`` and ``rubric``.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..models import AgentScore, CSSample
from ..prompting import (
    as_user,
    describe_principles,
    describe_request,
    json_only_instruction,
    parse_model,
)
from .base import ScorerAgent

_SYSTEM = (
    "You are a meticulous bilingual linguist evaluating code-switched utterances "
    "for a speech dataset. You judge one specific quality dimension at a time, "
    "strictly and consistently on a 0–10 scale, and you justify your score in one "
    "or two sentences."
)

_RESPONSE_SHAPE = '{"score": <number from 0 to 10>, "rationale": "<one or two sentences>"}'


class _ScoreOutput(BaseModel):
    """Parsed model output for one scoring call."""

    score: float
    rationale: str = ""


def _build_score_prompt(sample: CSSample, rubric: str) -> str:
    parts = [
        "Specification the utterance was written for:",
        describe_request(sample.request),
    ]
    principles = describe_principles(sample.request.principles)
    if principles:
        parts.append(principles)
    parts.extend(
        [
            "Utterance to score:",
            f"- Text: {sample.text}",
            f"- Translation: {sample.translation or '(none provided)'}",
            f"Scoring dimension — {rubric}",
            "Rate the utterance on the dimension above from 0 (very poor) to "
            "10 (excellent).",
            json_only_instruction(_RESPONSE_SHAPE),
        ]
    )
    return "\n\n".join(parts)


class _DimensionScorer(ScorerAgent):
    """Shared scoring path: prompt with the rubric, parse ``{score, rationale}``."""

    #: One-line description of the dimension, injected into the prompt.
    rubric: str = ""

    async def score(self, sample: CSSample) -> AgentScore:
        prompt = _build_score_prompt(sample, self.rubric)
        response = await self.llm.complete(as_user(prompt), system=_SYSTEM)
        out = parse_model(response.text, _ScoreOutput)
        score = max(0.0, min(10.0, out.score))  # clamp to AgentScore's 0–10 bound
        return AgentScore(agent=self.name, score=score, rationale=out.rationale)


class FluencyAgent(_DimensionScorer):
    """Verifies grammaticality and the absence of broken morphemes."""

    name = "FluencyAgent"
    rubric = (
        "Grammaticality and morphological integrity — is the utterance grammatical "
        "in both languages at the switch points, with no broken words, illformed "
        "morphology, or agreement errors? Penalize ungrammatical switches and "
        "broken morphemes."
    )


class NaturalnessAgent(_DimensionScorer):
    """Estimates pragmatic plausibility with a domain-conditioned LM."""

    name = "NaturalnessAgent"
    rubric = (
        "Naturalness and pragmatic plausibility — would a real speaker with this "
        "persona actually say this in this setting? Reward idiomatic, fluent phrasing "
        "and penalize stilted, translated-sounding, or implausible utterances."
    )


class CSRatioAgent(_DimensionScorer):
    """Checks whether the token-level language ratio matches the user target."""

    name = "CSRatioAgent"
    rubric = (
        "Code-switching ratio and structure — does the proportion of "
        "embedded-language (L2) tokens match the target ratio in the spec, and does "
        "the switching match the requested type and discourse function? Penalize "
        "under- or over-switching and the wrong switch structure."
    )


class SocialCultureAgent(_DimensionScorer):
    """Validates register, borrowed lexicon, and cultural appropriateness."""

    name = "SocialCultureAgent"
    rubric = (
        "Sociolinguistic appropriateness — are the register, borrowed lexicon, and "
        "cultural references appropriate for this persona, topic, and conversation "
        "type? Penalize mismatched register and culturally implausible choices."
    )
