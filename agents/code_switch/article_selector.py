"""ArticleSelectorAgent — pre-generation context enrichment.

Reads the full article pool from tool_context["news"], picks the single most
relevant article for the given persona + topic, and proposes a natural social
interaction frame. Results are written back into tool_context so GenerationAgent
can consume them.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent, fill_template, load_guide
from agents.helpers import _format_articles
from models import GenerationContext
from prompting import PromptParseError, as_user

log = logging.getLogger(__name__)

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."

_ARTICLE_SELECTOR_PROMPT = load_guide("prompts/article_selector.md")


class ArticleSelectorAgent(Agent):
    """Picks the best article and proposes an interaction frame before generation."""

    name = "ArticleSelectorAgent"

    async def select(self, ctx: GenerationContext) -> None:
        """Enrich ctx.tool_context with 'selected_article' and 'interaction_frame'."""
        news = (ctx.tool_context or {}).get("news") or {}
        articles = news.get("articles") or []
        if not articles:
            log.debug("%s: no articles in tool_context, skipping", self.name)
            return

        ch = ctx.request.character
        b = ctx.request.basic

        prompt = fill_template(
            _ARTICLE_SELECTOR_PROMPT,
            age=ch.age,
            gender=ch.gender,
            first_language=ch.first_language,
            second_language=ch.second_language,
            topic=b.topic,
            conversation_type=b.conversation_type,
            articles=_format_articles(articles),
        )

        def _validate(data: Any) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"{self.name}: expected JSON object, got {type(data).__name__}")
            idx = data.get("article_index")
            if not isinstance(idx, int) or not (0 <= idx < len(articles)):
                raise PromptParseError(
                    f"{self.name}: article_index {idx!r} out of range [0, {len(articles)})"
                )
            frame = data.get("frame")
            if not isinstance(frame, str) or not frame.strip():
                raise PromptParseError(f"{self.name}: 'frame' must be a non-empty string")
            return data

        data = await self._complete_with_retry(as_user(prompt), system=_SYSTEM, validate=_validate)

        selected = articles[data["article_index"]]
        ctx.tool_context["selected_article"] = selected
        ctx.tool_context["interaction_frame"] = data["frame"].strip()
        log.info(
            "%s: selected article %d/%d  frame=%r  title=%r",
            self.name,
            data["article_index"],
            len(articles),
            ctx.tool_context["interaction_frame"],
            selected.get("title", ""),
        )
