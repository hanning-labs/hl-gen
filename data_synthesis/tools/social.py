"""Social media tool provider — stub."""

from __future__ import annotations

from typing import Any

from ..models import GenerationContext


class SocialMediaTool:
    """Pulls colloquial, register-rich context from social media."""

    name = "social"

    async def fetch(self, ctx: GenerationContext) -> dict[str, Any]:
        # Later: query a social platform for posts related to the topic/persona.
        raise NotImplementedError
