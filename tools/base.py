"""Tool integration interface for external context providers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from models import GenerationContext


@runtime_checkable
class ToolProvider(Protocol):
    """A source of external context for generation.

    Each provider exposes a ``name`` and an async ``fetch`` returning a context
    dict, which the orchestrator merges into ``GenerationContext.tool_context``
    under that name.
    """

    name: str

    async def fetch(self, ctx: GenerationContext) -> dict[str, Any]:
        ...

