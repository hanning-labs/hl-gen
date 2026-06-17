"""GenerationAgent — the Generator (Table B.1)."""

from __future__ import annotations

from ..models import CSSample, GenerationContext
from .base import GeneratorAgent


class GenerationAgent(GeneratorAgent):
    """Produces a CS sentence given topic, persona, and matrix/embedded languages."""

    name = "GenerationAgent"

    async def generate(self, ctx: GenerationContext) -> CSSample:
        # Later: build a prompt from ctx.request (code-switching spec, persona,
        # basic settings, linguistic principles) plus ctx.tool_context, fold in
        # ctx.feedback on refinement rounds, then call self.llm.complete(...).
        raise NotImplementedError
