"""NewsAPI.org tool provider.

Fetches topical news articles from NewsAPI (https://newsapi.org/docs) to ground
generation with real-world context. Runs alongside :class:`CurrentsTool` under its
own ``tool_context`` key. Requires a free API key — set ``NEWS_API_KEY`` or pass
``api_key`` directly.

Endpoint is chosen at runtime: if the request topic is one of NewsAPI's seven
top-headlines categories (configs pass these directly), ``/v2/top-headlines`` is
used; otherwise the keyword ``/v2/everything`` search with a date window is used.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from dotenv import load_dotenv

from models import GenerationContext
from tools.currents import _ALL_CATEGORIES, _LANGUAGE_TO_ISO, _BODY_TRUNCATE

log = logging.getLogger(__name__)

_EVERYTHING_URL = "https://newsapi.org/v2/everything"
_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"

# Languages NewsAPI actually accepts (ISO-639-1). Anything else → omit the param.
_NEWSAPI_LANGS = {"ar", "de", "en", "es", "fr", "he", "it", "nl", "no", "pt", "ru", "sv", "ud", "zh"}

# NewsAPI's 7 valid top-headlines categories. Configs for this tool use these
# names directly as topics; anything else falls through to /everything keyword search.
_NEWSAPI_CATEGORIES = {"business", "entertainment", "general", "health", "science", "sports", "technology"}


class NewsAPITool:
    """Fetches NewsAPI.org articles relevant to the request topic."""

    name = "newsapi"

    def __init__(
        self,
        api_key: str | None = None,
        max_articles: int = 30,
        language: str = "en",
        categories: list[str] | None = None,
        news_types: list[str] | None = None,  # ponytail: accepted for signature-compat with CurrentsTool; NewsAPI has no equivalent
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("NEWS_API_KEY", "")
        self.max_articles = max_articles
        self.language = language
        self.categories = categories or _ALL_CATEGORIES

    def _fetch_sync(
        self,
        keyword: str,
        language: str | None,
        category: str | None,
        page_size: int,
        start_date: str | None,
        end_date: str | None,
        page: int,
    ) -> dict[str, Any]:
        if not self.api_key:
            log.warning("NEWS_API_KEY not set — skipping news fetch")
            return {"articles": [], "source": "newsapi", "error": "NEWS_API_KEY not set"}

        # top-headlines when the topic maps to a NewsAPI category; else keyword search.
        if category:
            url = _HEADLINES_URL
            # No q: category alone — adding the topic word as a query filters out nearly everything.
            params: dict[str, Any] = {"category": category, "pageSize": page_size, "page": page}
        else:
            url = _EVERYTHING_URL
            # sortBy=publishedAt (not relevancy) so the random date window actually changes
            # which articles rank first; relevancy would return the same top hits regardless.
            params = {"q": keyword, "pageSize": page_size, "sortBy": "publishedAt", "page": page}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date
        if language:
            params["language"] = language

        log.info(
            "Fetching NewsAPI articles  endpoint=%s  q=%r  lang=%s  category=%s  size=%d",
            url.rsplit("/", 1)[-1], keyword, language, category, page_size,
        )
        try:
            resp = requests.get(url, params=params, headers={"X-Api-Key": self.api_key}, timeout=10)
            data = resp.json()
        except Exception as exc:
            log.error("NewsAPI request failed  q=%r  error=%s", keyword, exc)
            return {"articles": [], "source": "newsapi", "error": str(exc)}

        if data.get("status") != "ok":
            msg = data.get("message") or data.get("code") or str(data)
            log.error("NewsAPI error  q=%r  status=%s  response=%s", keyword, resp.status_code, data)
            return {"articles": [], "source": "newsapi", "error": msg}

        articles = []
        for item in data.get("articles", [])[: self.max_articles]:
            description = (item.get("description") or "")[:_BODY_TRUNCATE]
            articles.append({
                "id": item.get("url", ""),  # NewsAPI has no article id; url is the stable key
                "title": item.get("title", ""),
                "body": description,
                "url": item.get("url", ""),
                "date": item.get("publishedAt", ""),
                "author": item.get("author", ""),
                "language": language or "",
                "category": [category] if category else [],
            })

        log.info("Fetched %d articles  q=%r", len(articles), keyword)
        return {
            "articles": articles,
            "source": "newsapi",
            "total_results": data.get("totalResults"),
        }

    async def fetch(self, ctx: GenerationContext) -> dict[str, Any]:
        character = getattr(ctx.request, "character", None)
        l1 = character.first_language if character else "english"
        iso = _LANGUAGE_TO_ISO.get(l1.lower(), self.language)
        language = iso if iso in _NEWSAPI_LANGS else None
        if language is None:
            log.warning("NewsAPI has no language %r (from L1 %r) — omitting language filter", iso, l1)

        topic_slug = ctx.request.topic
        category = topic_slug if topic_slug in _NEWSAPI_CATEGORIES else None  # None → /everything keyword search
        keyword = topic_slug.replace("_", " ")

        page_size = random.randint(min(5, self.max_articles), self.max_articles)
        # Free plan caps results at 100 total; pick a random page within that window
        # so repeated calls land on different articles instead of always page 1.
        page = random.randint(1, max(1, 100 // page_size))

        now = datetime.now(timezone.utc)
        end_dt = now - timedelta(days=random.randint(0, 22))
        start_dt = end_dt - timedelta(days=7)
        end_date = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
        start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%S")

        return await asyncio.to_thread(
            self._fetch_sync, keyword, language, category, page_size, start_date, end_date, page,
        )


def demo() -> None:
    """Offline self-check: no key/network. Verifies normalization + endpoint routing."""
    from types import SimpleNamespace

    payload = {
        "status": "ok",
        "totalResults": 1,
        "articles": [{
            "title": "Quantum breakthrough",
            "description": "A lab reports a new result.",
            "url": "https://example.com/a",
            "publishedAt": "2026-06-30T12:00:00Z",
            "author": "J. Doe",
        }],
    }
    calls: list[str] = []
    sent_params: list[dict] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        sent_params.append(params or {})
        assert headers and headers.get("X-Api-Key"), "API key must go in X-Api-Key header"
        return SimpleNamespace(status_code=200, json=lambda: payload)

    global requests
    real_requests = requests
    requests = SimpleNamespace(get=fake_get)  # type: ignore
    try:
        tool = NewsAPITool()  # key pulled from NEWS_API_KEY via load_dotenv()
        assert tool.api_key, "NEWS_API_KEY missing — set it in .env"

        def ctx_for(topic: str) -> GenerationContext:
            req = SimpleNamespace(
                topic=topic,
                character=SimpleNamespace(first_language="english"),
            )
            return GenerationContext(request=req)

        # Mapped topic → top-headlines, category only (no q — it would filter out everything).
        res = asyncio.run(tool.fetch(ctx_for("health")))
        assert calls[-1] == _HEADLINES_URL, calls
        assert "q" not in sent_params[-1], sent_params[-1]
        assert "page" in sent_params[-1], sent_params[-1]
        art = res["articles"][0]
        assert set(art) == {"id", "title", "body", "url", "date", "author", "language", "category"}
        assert art["body"] == "A lab reports a new result."
        assert art["language"] == "en"
        assert art["category"] == ["health"]
        assert res["source"] == "newsapi"

        # NewsAPI-native config names → top-headlines.
        for topic in ("business", "entertainment", "science", "sports", "technology", "general"):
            asyncio.run(tool.fetch(ctx_for(topic)))
            assert calls[-1] == _HEADLINES_URL, (topic, calls[-1])

        # Unmapped topic → everything, ranked by relevancy.
        asyncio.run(tool.fetch(ctx_for("my_weird_hobby")))
        assert calls[-1] == _EVERYTHING_URL, calls
        assert sent_params[-1]["sortBy"] == "publishedAt", sent_params[-1]
        assert "page" in sent_params[-1], sent_params[-1]

        # Missing key → graceful empty, no request.
        empty = NewsAPITool(api_key="x")
        empty.api_key = ""
        out = asyncio.run(empty.fetch(ctx_for("health")))
        assert out["articles"] == [] and "error" in out
    finally:
        requests = real_requests  # type: ignore

    print("newsapi demo: OK")


if __name__ == "__main__":
    demo()
