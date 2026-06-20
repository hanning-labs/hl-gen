"""Currents API tool provider.

Fetches topical news articles from the Currents API to ground generation
with real-world context. Requires a free API key from
https://currentsapi.services/en/register — set the env var
``CURRENTS_API_KEY`` or pass ``api_key`` directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

from models import GenerationContext

_BASE_URL = "https://api.currentsapi.services/v2/search"
_BODY_TRUNCATE = 800  # chars per description kept in context


class CurrentsTool:
    """Fetches Currents API articles relevant to the request topic."""

    name = "news"

    def __init__(
        self,
        api_key: str | None = None,
        max_articles: int = 30,
        language: str = "en",
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("CURRENTS_API_KEY", "")
        self.max_articles = max_articles
        self.language = language

    def _fetch_sync(self, topic: str) -> dict[str, Any]:
        if not self.api_key:
            log.warning("CURRENTS_API_KEY not set — skipping news fetch")
            return {"articles": [], "source": "currents", "error": "CURRENTS_API_KEY not set"}

        log.info("Fetching Currents articles  topic=%r  max=%d", topic, self.max_articles)
        try:
            resp = requests.get(
                _BASE_URL,
                params={
                    "keywords": topic,
                    "language": self.language,
                    "page_number": 1,
                    "page_size": self.max_articles,
                    "apiKey": self.api_key,
                },
                timeout=10,
            )
            data = resp.json()
        except Exception as exc:
            log.error("Currents request failed  topic=%r  error=%s", topic, exc)
            return {"articles": [], "source": "currents", "error": str(exc)}

        if data.get("status") != "ok":
            msg = data.get("message", "unknown error")
            log.error("Currents API error  topic=%r  message=%s", topic, msg)
            return {"articles": [], "source": "currents", "error": msg}

        articles = []
        for item in data.get("news", [])[: self.max_articles]:
            description = (item.get("description") or "")[:_BODY_TRUNCATE]
            articles.append({
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "body": description,
                "url": item.get("url", ""),
                "date": item.get("published", ""),
                "author": item.get("author", ""),
                "language": item.get("language", ""),
                "category": item.get("category", []),
            })
        log.info("Fetched %d articles  topic=%r", len(articles), topic)
        return {
            "articles": articles,
            "source": "currents",
            "page": data.get("page"),
            "next_cursor": data.get("next_cursor"),
        }

    async def fetch(self, ctx: GenerationContext) -> dict[str, Any]:
        return await asyncio.to_thread(self._fetch_sync, ctx.request.basic.topic)
