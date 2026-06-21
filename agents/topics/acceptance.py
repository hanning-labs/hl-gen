"""TopicsAcceptanceAgent — Sink for the topics pipeline."""

from __future__ import annotations

from llm.base import LLMClient
from models import CSSample, ScoreReport
from storage.base import SampleStore
from agents.base import SinkAgent


class TopicsAcceptanceAgent(SinkAgent):
    """Stores accepted topic samples; flattens request via model_dump()."""

    name = "TopicsAcceptanceAgent"

    def __init__(self, llm: LLMClient, store: SampleStore, *, name: str | None = None) -> None:
        super().__init__(llm, name=name)
        self.store = store

    async def accept(self, sample: CSSample, report: ScoreReport, tool_context: dict) -> None:
        sample.metadata.update(
            {
                "accepted_by": self.name,
                "final_score": report.final_score,
                "passed": report.passed,
                "scores": [s.model_dump() for s in report.scores],
                "spec": sample.request.model_dump(),
                "tool_context": tool_context,
            }
        )
        await self.store.save(sample, report)
