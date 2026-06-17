"""Claude (Anthropic) reference implementation of LLMClient — stub.

Construction is allowed so pipelines can be wired and inspected; the actual
API call in :meth:`complete` is implemented in a later pass (it will use the
``anthropic`` SDK, available via the ``claude`` optional dependency).
"""

from __future__ import annotations

from typing import Any

from .base import LLMResponse, Message

# Default to the latest, most capable Claude model. See the claude-api reference
# for current model ids before changing this.
DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeClient:
    """:class:`LLMClient` backed by the Anthropic API."""

    def __init__(self, *, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError("ClaudeClient.complete is implemented in a later pass.")
