"""File-backed reference :class:`SampleStore` (JSONL).

Each accepted sample is appended as one JSON line holding a stable id, a
timestamp, and the full ``sample`` + ``report`` dumps. File I/O is blocking, so
writes run in a worker thread under a lock — safe for the concurrent callers the
pipeline may produce. This is the simple reference store; the planned audio
corpus DB implements the same protocol later.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..models import CSSample, ScoreReport


class FileSampleStore:
    """Append-only JSONL store of accepted samples and their score reports."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _append(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    async def save(self, sample: CSSample, report: ScoreReport) -> str:
        """Persist ``sample`` + ``report`` as one JSONL record; return its id."""
        sample_id = uuid.uuid4().hex
        record = {
            "id": sample_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "sample": sample.model_dump(mode="json"),
            "report": report.model_dump(mode="json"),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)
        return sample_id

    def read_all(self) -> list[dict]:
        """Read every stored record (convenience for inspection/tests)."""
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
