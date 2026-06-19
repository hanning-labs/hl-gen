"""Guardian Open Platform tool provider.

Fetches topical news articles from The Guardian to ground generation with
real-world context. Requires a free developer API key from
https://open-platform.theguardian.com/member/register — set the env var
``GUARDIAN_API_KEY`` or pass ``api_key`` directly.

The ``"test"`` key works against the Guardian sandbox but returns limited
results and is rate-throttled; obtain a real key for production use.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from models import GenerationContext

_BASE_URL = "https://content.guardianapis.com/search"
_BODY_TRUNCATE = 800  # chars per article kept in context


class GuardianTool:
    """Fetches Guardian articles relevant to the request topic."""

    name = "news"

    def __init__(
        self,
        api_key: str | None = None,
        max_articles: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("GUARDIAN_API_KEY", "test")
        self.max_articles = max_articles

    def _fetch_sync(self, topic: str) -> dict[str, Any]:
        params = {
            "q": topic,
            "api-key": self.api_key,
            "page-size": str(self.max_articles),
            "show-fields": "bodyText",
            "order-by": "relevance",
        }
        url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            return {"articles": [], "source": "guardian", "error": str(exc)}

        results = data.get("response", {}).get("results", [])
        articles = []
        for r in results:
            body = (r.get("fields") or {}).get("bodyText", "")
            articles.append({
                "title": r.get("webTitle", ""),
                "body": body[:_BODY_TRUNCATE],
                "url": r.get("webUrl", ""),
                "date": r.get("webPublicationDate", ""),
            })
        return {"articles": articles, "source": "guardian"}

    async def fetch(self, ctx: GenerationContext) -> dict[str, Any]:
        return await asyncio.to_thread(self._fetch_sync, ctx.request.basic.topic)
