"""GDELT Doc 2.0 API tool provider.

Uses the ``gdeltdoc`` library (https://github.com/alex9smith/gdelt-doc-api)
which sets a proper User-Agent header and handles query-string construction
correctly. Install via the ``tools`` optional dependency group:

    pip install -e ".[tools]"

GDELT does not return article body text — only titles, URLs, and metadata.
The fetch result therefore contains headlines rather than full articles; this
is still useful as topical grounding for the generator.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from models import GenerationContext

# On 429, retry with escalating waits (5 s → 10 s → 15 s; total ≤ 30 s).
_RETRY_DELAYS = (5.0, 10.0, 15.0)


class GDELTTool:
    """Fetches GDELT article headlines relevant to the request topic."""

    name = "news"

    def __init__(self, max_articles: int = 5, timespan: str = "1w") -> None:
        self.max_articles = max_articles
        self.timespan = timespan

    def _fetch_sync(self, topic: str) -> dict[str, Any]:
        def _error(exc: Exception) -> dict[str, Any]:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            msg = type(exc).__name__
            if status:
                msg = f"{msg} (HTTP {status})"
            return {"articles": [], "source": "gdelt", "error": msg}

        try:
            from gdeltdoc import Filters, GdeltDoc
            from gdeltdoc.errors import RateLimitError
        except ImportError as exc:
            raise ImportError(
                "GDELTTool requires the 'tools' optional dependencies. "
                'Install them with: pip install -e ".[tools]"'
            ) from exc

        f = Filters(
            keyword=topic,
            language="english",
            timespan=self.timespan,
            num_records=self.max_articles,
        )

        def _search() -> Any:
            return GdeltDoc().article_search(f)

        try:
            df = _search()
        except RateLimitError:
            df = None
            for delay in _RETRY_DELAYS:
                time.sleep(delay)
                try:
                    df = _search()
                    break
                except RateLimitError:
                    continue
                except Exception as exc:
                    return _error(exc)
            if df is None:
                return {"articles": [], "source": "gdelt", "error": "RateLimitError: retries exhausted"}
        except Exception as exc:
            return _error(exc)

        if df.empty:
            return {"articles": [], "source": "gdelt"}

        articles = []
        for _, row in df.head(self.max_articles).iterrows():
            articles.append({
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "date": row.get("seendate", ""),
                "domain": row.get("domain", ""),
            })
        return {"articles": articles, "source": "gdelt"}

    async def fetch(self, ctx: GenerationContext) -> dict[str, Any]:
        return await asyncio.to_thread(self._fetch_sync, ctx.request.basic.topic)
