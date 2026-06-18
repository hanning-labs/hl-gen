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

    async def accept(self, sample: CSSample, report: ScoreReport) -> None:
        cs = sample.request.code_switching
        ch = sample.request.character
        b = sample.request.basic

        # Attach final scores + provenance, plus a flattened snapshot of the spec
        # (the internal request has no downstream visibility once generated).
        sample.metadata.update(
            {
                "accepted_by": self.name,
                "final_score": report.final_score,
                "passed": report.passed,
                "scores": [s.model_dump() for s in report.scores],
                "spec": {
                    "first_language": ch.first_language,
                    "second_language": ch.second_language,
                    "age": ch.age,
                    "gender": ch.gender,
                    "perspective": b.perspective,
                    "tense": b.tense,
                    "topic": b.topic,
                    "conversation_type": b.conversation_type,
                    "cs_type": cs.type.value,
                    "cs_function": cs.function,
                    "cs_ratio": cs.ratio,
                },
            }
        )

        await self.store.save(sample, report)
