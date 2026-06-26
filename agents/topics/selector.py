"""TopicArticleSelectorAgent — pre-generation context enrichment for the topics pipeline.

Like ArticleSelectorAgent but without a persona — picks the article most relevant
to the topic and proposes an angle/framing for the requested style.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.helpers import _format_articles
from agents.base import Agent
from models import GenerationContext
from prompting import PromptParseError, as_user

log = logging.getLogger(__name__)

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."


class TopicArticleSelectorAgent(Agent):
    """Picks the best article and proposes a style-appropriate frame before generation."""

    name = "TopicArticleSelectorAgent"

    async def select(self, ctx: GenerationContext) -> None:
        """Enrich ctx.tool_context with 'selected_article' and 'interaction_frame'."""
        news = (ctx.tool_context or {}).get("currents_api") or {}
        articles = news.get("articles") or []
        if not articles:
            log.debug("%s: no articles in tool_context, skipping", self.name)
            return

        topic = ctx.request.topic
        style = ctx.request.style

        def _validate_frame(data: Any) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"{self.name}: expected JSON object, got {type(data).__name__}")
            frame = data.get("frame")
            if not isinstance(frame, str) or not frame.strip():
                raise PromptParseError(f"{self.name}: 'frame' must be a non-empty string")
            return data

        if len(articles) == 1:
            selected = articles[0]
            ctx.tool_context["selected_article"] = selected
            title = (selected.get("title") or "").strip()
            body = (selected.get("body") or "").strip()
            frame_prompt = (
                f"Topic: {topic}\nStyle: {style}\n\n"
                f"Article: {title}\n{body}\n\n"
                "Propose a specific engagement frame (1 sentence) telling the generator exactly HOW "
                f"to write a {style!r} piece using this article — the angle, approach, and what to focus on.\n"
                'Respond with: {"frame": "<one sentence>"}'
            )
            data = await self._complete_with_retry(as_user(frame_prompt), system=_SYSTEM, validate=_validate_frame)
            article_idx = 0
        else:
            combined_prompt = (
                "You are selecting the most relevant news article for a topic-focused text generation task.\n\n"
                f"Topic: {topic}\n"
                f"Requested style: {style}\n\n"
                f"Articles:\n{_format_articles(articles)}\n\n"
                "Pick the article most relevant to the topic and best suited for the requested style.\n"
                "Also propose a specific engagement frame (1 sentence) that tells the generator exactly HOW to write "
                f"a {style!r} piece using this article — the angle, approach, and what to focus on.\n"
                'Respond with: {"article_index": <integer>, "frame": "<one sentence>"}'
            )

            def _validate_combined(data: Any) -> dict:
                _validate_frame(data)
                idx = data.get("article_index")
                if not isinstance(idx, int) or not (0 <= idx < len(articles)):
                    raise PromptParseError(
                        f"{self.name}: article_index {idx!r} out of range [0, {len(articles)})"
                    )
                return data

            data = await self._complete_with_retry(as_user(combined_prompt), system=_SYSTEM, validate=_validate_combined)
            article_idx = data["article_index"]
            selected = articles[article_idx]
            ctx.tool_context["selected_article"] = selected

        ctx.tool_context["interaction_frame"] = data["frame"].strip()
        log.info(
            "%s: article %d/%d  frame=%r  title=%r",
            self.name,
            article_idx,
            len(articles),
            ctx.tool_context["interaction_frame"],
            selected.get("title", ""),
        )
