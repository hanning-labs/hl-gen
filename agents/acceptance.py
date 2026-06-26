"""AcceptanceAgent — the Sink (Table B.1)."""

from __future__ import annotations

from llm.base import LLMClient
from models import CSSample, ScoreReport
from storage.base import SampleStore
from .base import SinkAgent


class AcceptanceAgent(SinkAgent):
    """Stores accepted samples and logs their metadata via a SampleStore."""

    name = "AcceptanceAgent"

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
