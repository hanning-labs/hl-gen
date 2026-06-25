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
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

from models import GenerationContext

_BASE_URL = "https://api.currentsapi.services/v2/search"
_BODY_TRUNCATE = 2500  # chars per description kept in context


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
        page_size_min: int = 10,
        page_size_max: int = 30,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("CURRENTS_API_KEY", "")
        self.max_articles = max_articles
        self.language = language
        self.categories = categories or _ALL_CATEGORIES
        self.news_types = news_types or list(_CONTENT_TYPE.keys())
        self.page_size_min = page_size_min
        self.page_size_max = page_size_max

    def _fetch_sync(
        self,
        topic: str,
        language: str,
        category: str | None = None,
        content_type: int | None = None,
        page_number: int = 1,
        page_size: int = 30,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            log.warning("CURRENTS_API_KEY not set — skipping news fetch")
            return {"articles": [], "source": "currents", "error": "CURRENTS_API_KEY not set"}

        log.info(
            "Fetching Currents articles  topic=%r  lang=%s  category=%s  type=%s  page=%d  size=%d  start=%s  end=%s",
            topic, language, category, content_type, page_number, page_size, start_date, end_date,
        )
        params: dict[str, Any] = {
            "keywords": topic,
            "language": language,
            "page_number": page_number,
            "page_size": page_size,
            "apiKey": self.api_key,
        }
        if category:
            params["category"] = category
        if content_type is not None:
            params["type"] = content_type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
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
            msg = data.get("message") or data.get("error") or str(data)
            log.error("Currents API error  topic=%r  status=%s  response=%s", topic, resp.status_code, data)
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
        # If deep pagination returned nothing, retry at page 1.
        if not articles and page_number > 1:
            log.info("Page %d returned 0 articles — retrying at page 1", page_number)
            return self._fetch_sync(topic, language, category, content_type, 1, page_size, start_date, end_date)

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
        # Use the request topic as category if it matches a known API category;
        # otherwise fall back to a random category.
        topic_slug = ctx.request.topic
        category = topic_slug if topic_slug in _ALL_CATEGORIES else random.choice(self.categories)
        content_type = _CONTENT_TYPE[random.choice(self.news_types)]

        # Humanize the keyword (slug → space-separated) so the API actually matches articles.
        keyword = topic_slug.replace("_", " ")

        # Random pagination within the offset guardrail: (page_number-1)*page_size <= 1000
        page_size = random.randint(self.page_size_min, self.page_size_max)
        max_page = min(180, 1000 // page_size)
        page_number = random.randint(1, max_page)

        # Date window: 14-day window (API max is 15), position shifts by age
        # age 18 → ends today; age 78+ → ends 15 days ago
        age = int(getattr(character, "age", 35) or 35)
        end_offset = int(min(max(age - 18, 0), 60) / 60 * 15)
        now = datetime.now(timezone.utc)
        end_dt = now - timedelta(days=end_offset)
        start_dt = end_dt - timedelta(days=14)
        end_date = end_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

        return await asyncio.to_thread(
            self._fetch_sync, keyword, language, category, content_type,
            page_number, page_size, start_date, end_date,
        )
