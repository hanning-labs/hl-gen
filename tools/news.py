"""News API tool provider — stub."""

from __future__ import annotations

from typing import Any

from models import GenerationContext


class NewsAPITool:
    """Pulls topical context from a news source to ground generation."""

    name = "news"

    async def fetch(self, ctx: GenerationContext) -> dict[str, Any]:
        # Later: query a news API for items related to ctx.request.basic.topic.
        raise NotImplementedError
