"""Control-flow tests for SynthesisPipeline using mock agents.

These exercise the orchestrator loop independently of the (unimplemented) real
agent bodies: the first-round accept path, the refine-then-accept path, and the
exhausted-rounds path. Run with: ``pytest`` (needs the ``dev`` extra).
"""

from __future__ import annotations

import pytest

from config import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from agents.base import (
    EditorAgent,
    GeneratorAgent,
    ReducerAgent,
    ScorerAgent,
    SinkAgent,
)
from models import (
    AgentScore,
    CSSample,
    GenerationContext,
    RefinementFeedback,
    ScoreReport,
)
from orchestrator import SynthesisPipeline


def make_request(*, threshold: float = 8.0, max_rounds: int = 3) -> SynthesisRequest:
    """A concrete Cantonese-L1 / English-L2 'Movie' request (cf. the diagram)."""
    return SynthesisRequest(
        code_switching=CodeSwitchingSpec(
            type=CodeSwitchType.INTRA_SENTENTIAL, function="emphasis", ratio=0.3
        ),
        character=CharacterSetting(
            first_language="Cantonese", second_language="English", age=28, gender="F"
        ),
        basic=BasicSetting(
            perspective="first-person",
            tense="present",
            topic="Movie",
            conversation_type="casual",
        ),
        score_threshold=threshold,
        max_refinement_rounds=max_rounds,
    )


# --- Mock agents: override only the abstract method; skip the llm dependency. ---


class MockGenerator(GeneratorAgent):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, ctx: GenerationContext) -> CSSample:
        self.calls += 1
        return CSSample(text=f"sample-{self.calls}", request=ctx.request)


class ConstantScorer(ScorerAgent):
    def __init__(self, value: float) -> None:
        self.value = value

    async def score(self, sample: CSSample) -> AgentScore:
        return AgentScore(agent="constant", score=self.value)


class RisingScorer(ScorerAgent):
    """Low score first, high after one refinement round."""

    def __init__(self) -> None:
        self.calls = 0

    async def score(self, sample: CSSample) -> AgentScore:
        self.calls += 1
        return AgentScore(agent="rising", score=3.0 if self.calls == 1 else 9.0)


class MeanReducer(ReducerAgent):
    def __init__(self) -> None:
        pass

    async def summarize(self, scores: list[AgentScore], *, threshold: float) -> ScoreReport:
        final = sum(s.score for s in scores) / len(scores)
        return ScoreReport(scores=scores, final_score=final, passed=final >= threshold)


class CountingRefiner(EditorAgent):
    def __init__(self) -> None:
        self.calls = 0

    async def refine(self, sample: CSSample, report: ScoreReport) -> RefinementFeedback:
        self.calls += 1
        return RefinementFeedback(failures=["below threshold"], suggestions="raise quality")


class RecordingSink(SinkAgent):
    def __init__(self) -> None:
        self.saved: list[tuple[CSSample, ScoreReport]] = []

    async def accept(self, sample: CSSample, report: ScoreReport) -> None:
        self.saved.append((sample, report))


def build(scorer: ScorerAgent) -> tuple[SynthesisPipeline, MockGenerator, CountingRefiner, RecordingSink]:
    gen, refiner, sink = MockGenerator(), CountingRefiner(), RecordingSink()
    pipe = SynthesisPipeline(
        generator=gen,
        scorers=[scorer],
        summarizer=MeanReducer(),
        refiner=refiner,
        acceptor=sink,
    )
    return pipe, gen, refiner, sink


@pytest.mark.asyncio
async def test_accept_on_first_round() -> None:
    pipe, gen, refiner, sink = build(ConstantScorer(9.0))
    out = await pipe.run(make_request())
    assert out is not None
    assert gen.calls == 1
    assert refiner.calls == 0
    assert len(sink.saved) == 1


@pytest.mark.asyncio
async def test_refine_then_accept() -> None:
    pipe, gen, refiner, sink = build(RisingScorer())
    out = await pipe.run(make_request())
    assert out is not None
    assert gen.calls == 2  # one regeneration after refinement
    assert refiner.calls == 1
    assert len(sink.saved) == 1


@pytest.mark.asyncio
async def test_exhausts_rounds_returns_none() -> None:
    pipe, gen, refiner, sink = build(ConstantScorer(1.0))
    out = await pipe.run(make_request(max_rounds=2))
    assert out is None
    assert gen.calls == 2
    assert refiner.calls == 2
    assert sink.saved == []
