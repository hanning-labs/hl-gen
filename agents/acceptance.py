"""AcceptanceAgent — the Sink (Table B.1)."""

from __future__ import annotations

from ..llm.base import LLMClient
from ..models import CSSample, ScoreReport
from ..storage.base import SampleStore
from .base import SinkAgent


class AcceptanceAgent(SinkAgent):
    """Stores accepted samples and logs their metadata via a SampleStore."""

    name = "AcceptanceAgent"

    def __init__(self, llm: LLMClient, store: SampleStore, *, name: str | None = None) -> None:
        super().__init__(llm, name=name)
        self.store = store

    async def accept(self, sample: CSSample, report: ScoreReport) -> None:
        # Later: attach final scores + provenance metadata to the sample and
        # persist it via self.store.save(sample, report).
        raise NotImplementedError
