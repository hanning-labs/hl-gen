"""Tool integration interface (MCP-style) + custom-hook extension point.

The framework's "Tool Integration" block lets generation draw on external
context (news, social media, …). Each source is a :class:`ToolProvider`,
modeled on MCP tools; user extensions plug in as :class:`CustomHook`s.
"""

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


@runtime_checkable
class CustomHook(Protocol):
    """User extension point (the "Custom Hooks" block).

    A hook receives the generation context and returns a (possibly enriched)
    context. Concrete hook points around the loop are defined in a later pass.
    """

    name: str

    async def __call__(self, ctx: GenerationContext) -> GenerationContext:
        ...
