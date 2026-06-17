"""SynthesisPipeline — the LinguaMaster generate / score / refine loop.

This is the one place with real control flow (not a stub): it wires the eight
framework agents and drives the closed loop —

    tools → generate → score (concurrently) → summarize → threshold?
        ├─ pass → accept (Sink) and return the sample
        └─ fail → refine (Editor) → re-prompt the generator

up to ``request.max_refinement_rounds`` attempts.
"""

from __future__ import annotations

import asyncio

from .agents.acceptance import AcceptanceAgent
from .agents.base import (
    EditorAgent,
    GeneratorAgent,
    ReducerAgent,
    ScorerAgent,
    SinkAgent,
)
from .agents.generation import GenerationAgent
from .agents.refiner import RefinerAgent
from .agents.scorers import (
    CSRatioAgent,
    FluencyAgent,
    NaturalnessAgent,
    SocialCultureAgent,
)
from .agents.summarize import SummarizeAgent
from .config import SynthesisRequest
from .llm.base import LLMClient
from .models import CSSample, GenerationContext
from .storage.base import SampleStore
from .tools.base import ToolProvider


class SynthesisPipeline:
    """Runs the closed-loop synthesis for a single request."""

    def __init__(
        self,
        *,
        generator: GeneratorAgent,
        scorers: list[ScorerAgent],
        summarizer: ReducerAgent,
        refiner: EditorAgent,
        acceptor: SinkAgent,
        tools: list[ToolProvider] | None = None,
    ) -> None:
        self.generator = generator
        self.scorers = scorers
        self.summarizer = summarizer
        self.refiner = refiner
        self.acceptor = acceptor
        self.tools = tools or []

    async def _gather_tool_context(self, ctx: GenerationContext) -> dict:
        """Fetch context from every tool provider, keyed by provider name."""
        results = await asyncio.gather(*(t.fetch(ctx) for t in self.tools))
        return {provider.name: result for provider, result in zip(self.tools, results)}

    async def run(self, request: SynthesisRequest) -> CSSample | None:
        """Synthesize one sample.

        Returns the accepted :class:`CSSample`, or ``None`` if no attempt
        cleared the threshold within ``request.max_refinement_rounds``.
        """
        ctx = GenerationContext(request=request)
        ctx.tool_context = await self._gather_tool_context(ctx)

        for _ in range(request.max_refinement_rounds):
            sample = await self.generator.generate(ctx)

            scores = await asyncio.gather(*(s.score(sample) for s in self.scorers))
            report = await self.summarizer.summarize(
                list(scores), threshold=request.score_threshold
            )

            if report.passed:
                await self.acceptor.accept(sample, report)
                return sample

            ctx.feedback = await self.refiner.refine(sample, report)

        return None


def build_default_pipeline(
    llm: LLMClient,
    store: SampleStore,
    *,
    tools: list[ToolProvider] | None = None,
) -> SynthesisPipeline:
    """Assemble the eight standard LinguaMaster agents into a pipeline."""
    return SynthesisPipeline(
        generator=GenerationAgent(llm),
        scorers=[
            FluencyAgent(llm),
            NaturalnessAgent(llm),
            CSRatioAgent(llm),
            SocialCultureAgent(llm),
        ],
        summarizer=SummarizeAgent(llm),
        refiner=RefinerAgent(llm),
        acceptor=AcceptanceAgent(llm, store),
        tools=tools,
    )
