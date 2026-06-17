"""SummarizeAgent — the Reducer (Table B.1)."""

from __future__ import annotations

from ..models import AgentScore, ScoreReport
from .base import ReducerAgent


class SummarizeAgent(ReducerAgent):
    """Normalizes individual scores and computes the weighted mean S_final."""

    name = "SummarizeAgent"

    async def summarize(self, scores: list[AgentScore], *, threshold: float) -> ScoreReport:
        # Later: normalize each score, compute the weighted mean S_final using
        # the scorers' weights, and set passed = (S_final >= threshold).
        raise NotImplementedError
