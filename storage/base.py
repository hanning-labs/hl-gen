"""Persistence interface for accepted samples.

The :class:`~agents.acceptance.AcceptanceAgent` (the pipeline Sink) writes here.
:class:`~storage.file_store.FileSampleStore` is the shipped implementation
(append-only JSONL); further stores (DB, the planned audio corpus DB) implement
this same protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from models import CSSample, ScoreReport


@runtime_checkable
class SampleStore(Protocol):
    """Where accepted samples and their score reports are persisted."""

    async def save(self, sample: CSSample, report: ScoreReport) -> str:
        """Persist ``sample`` + ``report`` and return a stable id for it."""
        ...
