"""Scorer agents (Table B.1): Fluency, Naturalness, CSRatio, SocialCulture.

Each scores one quality dimension on a 0–10 scale; the SummarizeAgent combines
them into S_final. Scoring rubrics and prompts are implemented in a later pass.
"""

from __future__ import annotations

from ..models import AgentScore, CSSample
from .base import ScorerAgent


class FluencyAgent(ScorerAgent):
    """Verifies grammaticality and the absence of broken morphemes."""

    name = "FluencyAgent"

    async def score(self, sample: CSSample) -> AgentScore:
        raise NotImplementedError


class NaturalnessAgent(ScorerAgent):
    """Estimates pragmatic plausibility with a domain-conditioned LM."""

    name = "NaturalnessAgent"

    async def score(self, sample: CSSample) -> AgentScore:
        raise NotImplementedError


class CSRatioAgent(ScorerAgent):
    """Checks whether the token-level language ratio matches the user target."""

    name = "CSRatioAgent"

    async def score(self, sample: CSSample) -> AgentScore:
        raise NotImplementedError


class SocialCultureAgent(ScorerAgent):
    """Validates register, borrowed lexicon, and cultural appropriateness."""

    name = "SocialCultureAgent"

    async def score(self, sample: CSSample) -> AgentScore:
        raise NotImplementedError
