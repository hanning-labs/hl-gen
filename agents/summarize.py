"""SummarizeAgent — the Reducer (Table B.1)."""

from __future__ import annotations

from llm.base import LLMClient
from models import AgentScore, ScoreReport
from .base import ReducerAgent


class SummarizeAgent(ReducerAgent):
    """Combines per-scorer verdicts into S_final and the accept/reject decision.

    Deterministic — no model call. ``final_score`` is the weighted mean of the
    scorers' 0–10 scores and ``passed`` is ``final_score >= threshold`` (the
    threshold is on the same 0–10 scale as ``SynthesisRequest.score_threshold``).
    Weights default to equal; pass ``weights`` (agent name -> weight) to override
    — unlisted agents default to 1.0.

    ``min_dimension_score`` (optional) additionally requires every individual
    scorer to reach that floor — a high mean can no longer mask one dimension
    failing outright.
    """

    name = "SummarizeAgent"

    def __init__(
        self,
        llm: LLMClient,
        *,
        name: str | None = None,
        weights: dict[str, float] | None = None,
        min_dimension_score: float | None = None,
    ) -> None:
        super().__init__(llm, name=name)
        self.weights = weights or {}
        self.min_dimension_score = min_dimension_score

    async def summarize(self, scores: list[AgentScore], *, threshold: float) -> ScoreReport:
        if not scores:
            return ScoreReport(scores=[], final_score=0.0, passed=False)

        weighted = sum(self.weights.get(s.agent, 1.0) * s.score for s in scores)
        total_weight = sum(self.weights.get(s.agent, 1.0) for s in scores)
        final = weighted / total_weight if total_weight else 0.0

        passed = final >= threshold
        if passed and self.min_dimension_score is not None:
            passed = min(s.score for s in scores) >= self.min_dimension_score

        return ScoreReport(scores=scores, final_score=final, passed=passed)
