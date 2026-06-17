"""RefinerAgent — the Editor (Table B.1)."""

from __future__ import annotations

from ..models import CSSample, RefinementFeedback, ScoreReport
from .base import EditorAgent


class RefinerAgent(EditorAgent):
    """Receives failure explanations and rewrites / re-prompts the generator."""

    name = "RefinerAgent"

    async def refine(self, sample: CSSample, report: ScoreReport) -> RefinementFeedback:
        # Later: gather the sub-threshold scorers' rationales and turn them into
        # concrete rewrite guidance for the next generation attempt.
        raise NotImplementedError
