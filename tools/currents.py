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
import random
from typing import Any

import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

from models import GenerationContext

_BASE_URL = "https://api.currentsapi.services/v2/search"
_BODY_TRUNCATE = 800  # chars per description kept in context


_CONTENT_TYPE = {"news": 1, "articles": 2, "discussion": 3}

_ALL_CATEGORIES = [
    "general", "society", "science_technology", "politics_government",
    "economy_business_finance", "arts_culture_entertainment", "lifestyle_leisure",
    "human_interest", "sport", "crime_law_justice", "education",
    "environment", "labour", "health", "automotive", "real_estate",
]

# ISO 639-1 codes for the Currents API language filter.
# Varieties that share a writing system (e.g. Cantonese / Mandarin) map to the
# same code; unknown names fall back to the instance default.
_LANGUAGE_TO_ISO: dict[str, str] = {
    "afrikaans": "af",
    "arabic": "ar",
    "azerbaijani": "az",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "cantonese": "zh",
    "catalan": "ca",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "kazakh": "kk",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "mandarin": "zh",
    "marathi": "mr",
    "mongolian": "mn",
    "norwegian": "no",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovene": "sl",
    "somali": "so",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
    "welsh": "cy",
    "yoruba": "yo",
    "zulu": "zu",
}


class CurrentsTool:
    """Fetches Currents API articles relevant to the request topic."""

    name = "currents_api"

    def __init__(
        self,
        api_key: str | None = None,
        max_articles: int = 30,
        language: str = "en",
        categories: list[str] | None = None,
        news_types: list[str] | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("CURRENTS_API_KEY", "")
        self.max_articles = max_articles
        self.language = language
        self.categories = categories or _ALL_CATEGORIES
        self.news_types = news_types or list(_CONTENT_TYPE.keys())

    def _fetch_sync(
        self,
        topic: str,
        language: str,
        category: str | None = None,
        content_type: int | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            log.warning("CURRENTS_API_KEY not set — skipping news fetch")
            return {"articles": [], "source": "currents", "error": "CURRENTS_API_KEY not set"}

        log.info(
            "Fetching Currents articles  topic=%r  lang=%s  category=%s  type=%s  max=%d",
            topic, language, category, content_type, self.max_articles,
        )
        params: dict[str, Any] = {
            "keywords": topic,
            "language": language,
            "page_number": 1,
            "page_size": self.max_articles,
            "apiKey": self.api_key,
        }
        if category:
            params["category"] = category
        if content_type is not None:
            params["type"] = content_type
        try:
            resp = requests.get(
                _BASE_URL,
                params=params,
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
        character = getattr(ctx.request, "character", None)
        l1 = character.first_language if character else "english"
        language = _LANGUAGE_TO_ISO.get(l1.lower(), self.language)
        if language == self.language and l1.lower() not in _LANGUAGE_TO_ISO:
            log.warning("No ISO mapping for L1 %r — falling back to %r", l1, self.language)
        category = random.choice(self.categories)
        content_type = _CONTENT_TYPE[random.choice(self.news_types)]
        return await asyncio.to_thread(
            self._fetch_sync, ctx.request.topic, language, category, content_type
        )
