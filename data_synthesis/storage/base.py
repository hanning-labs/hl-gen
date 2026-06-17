"""Persistence interface for accepted samples.

The :class:`~code_switch.data_synthesis.agents.acceptance.AcceptanceAgent`
(the pipeline Sink) writes here. Concrete stores (filesystem, DB, the planned
audio corpus DB) implement this protocol in a later pass.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import CSSample, ScoreReport


@runtime_checkable
class SampleStore(Protocol):
    """Where accepted samples and their score reports are persisted."""

    async def save(self, sample: CSSample, report: ScoreReport) -> str:
        """Persist ``sample`` + ``report`` and return a stable id for it."""
        ...
