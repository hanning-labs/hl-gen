"""Provider-agnostic LLM client interface.

Agents code against the :class:`LLMClient` protocol rather than any concrete
SDK, so the backend (Claude, others) stays swappable. The reference backend is
:class:`~llm.claude.ClaudeClient`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat-turn message."""

    role: str  # "user" | "assistant"
    content: str


class LLMResponse(BaseModel):
    """Normalized completion result returned by any LLMClient."""

    text: str
    model: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None  # chain-of-thought trace, stripped from `text`
    raw: Any = None  # provider-native response object, for debugging


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async chat-completion surface the agents depend on."""

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a completion for ``messages`` under an optional ``system`` prompt."""
        ...
