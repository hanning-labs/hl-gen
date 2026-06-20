"""Control-flow tests for SynthesisPipeline using mock agents.

These exercise the orchestrator loop independently of the (unimplemented) real
agent bodies: the first-round accept path, the refine-then-accept path, and the
exhausted-rounds path. Run with: ``pytest`` (needs the ``dev`` extra).
"""

from __future__ import annotations

import pytest
import yaml

from batch import BatchConfig, BatchRun, LocalClientConfig, run_batch
from llm.local import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL as LOCAL_DEFAULT_MODEL
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


def test_local_client_config_defaults() -> None:
    cfg = LocalClientConfig()
    assert cfg.model == LOCAL_DEFAULT_MODEL
    assert cfg.device is None
    assert cfg.dtype == "auto"
    assert cfg.max_new_tokens == DEFAULT_MAX_NEW_TOKENS


def test_batch_config_client_defaults() -> None:
    cfg = BatchConfig.model_validate({"n": 5})
    assert isinstance(cfg.client, LocalClientConfig)
    assert cfg.client.model == LOCAL_DEFAULT_MODEL


def test_batch_config_client_from_yaml() -> None:
    raw = yaml.safe_load("""
n: 5
client:
  model: Qwen/Qwen2.5-14B-Instruct
  device: cuda
  dtype: bfloat16
  max_new_tokens: 512
""")
    cfg = BatchConfig.model_validate(raw)
    assert cfg.client.model == "Qwen/Qwen2.5-14B-Instruct"
    assert cfg.client.device == "cuda"
    assert cfg.client.dtype == "bfloat16"
    assert cfg.client.max_new_tokens == 512


def test_count_existing_empty(tmp_path) -> None:
    from batch import count_existing
    assert count_existing(str(tmp_path / "out.jsonl")) == 0


def test_count_existing_with_records(tmp_path) -> None:
    import json
    from batch import count_existing
    out = tmp_path / "out.jsonl"
    records = [{"id": str(i), "saved_at": "2026-01-01T00:00:00+00:00", "sample": {}, "report": {}} for i in range(5)]
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert count_existing(str(out)) == 5


def test_resume_skips_already_done(tmp_path) -> None:
    import json
    from batch import count_existing, BatchConfig
    out = tmp_path / "out.jsonl"
    records = [{"id": str(i), "saved_at": "2026-01-01T00:00:00+00:00", "sample": {}, "report": {}} for i in range(3)]
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    cfg = BatchConfig.model_validate({"n": 10, "output": str(out)})
    already_done = count_existing(cfg.output)
    remaining = cfg.n - already_done
    assert already_done == 3
    assert remaining == 7


def test_batch_config_seed_default_is_none() -> None:
    cfg = BatchConfig.model_validate({"n": 1})
    assert cfg.seed is None


def test_batch_config_seed_from_yaml() -> None:
    raw = yaml.safe_load("n: 5\nseed: 42")
    cfg = BatchConfig.model_validate(raw)
    assert cfg.seed == 42


def test_seeded_runs_produce_same_requests() -> None:
    from batch import sample_request
    import random

    cfg = BatchConfig.model_validate({"n": 10, "seed": 99})
    rng_a = random.Random(cfg.seed)
    rng_b = random.Random(cfg.seed)
    requests_a = [sample_request(cfg, rng_a) for _ in range(cfg.n)]
    requests_b = [sample_request(cfg, rng_b) for _ in range(cfg.n)]
    assert [r.model_dump() for r in requests_a] == [r.model_dump() for r in requests_b]


def test_unseeded_runs_differ() -> None:
    from batch import sample_request
    import random

    cfg = BatchConfig.model_validate({"n": 10})
    requests_a = [sample_request(cfg, random.Random()) for _ in range(cfg.n)]
    requests_b = [sample_request(cfg, random.Random()) for _ in range(cfg.n)]
    assert [r.model_dump() for r in requests_a] != [r.model_dump() for r in requests_b]


def test_local_client_config_batch_defaults() -> None:
    cfg = LocalClientConfig()
    assert cfg.max_batch_size == 1
    assert cfg.batch_timeout_sec == 0.02


def test_local_client_config_batch_from_yaml() -> None:
    raw = yaml.safe_load("max_batch_size: 4\nbatch_timeout_sec: 0.05")
    cfg = LocalClientConfig.model_validate(raw)
    assert cfg.max_batch_size == 4
    assert cfg.batch_timeout_sec == 0.05


def test_local_client_config_model_dump_matches_local_client_signature() -> None:
    """model_dump() keys must match LocalClient.__init__ parameter names exactly."""
    from llm.local import LocalClient
    import inspect
    sig_params = set(inspect.signature(LocalClient.__init__).parameters) - {"self"}
    cfg_keys = set(LocalClientConfig().model_dump().keys())
    assert cfg_keys == sig_params


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


@pytest.mark.asyncio
async def test_run_batch_returns_batch_run() -> None:
    pipe, _, _, _ = build(ConstantScorer(9.0))
    cfg = BatchConfig.model_validate({"n": 3, "max_concurrent": 3})
    result = await run_batch(cfg, pipe, seed=0)
    assert isinstance(result, BatchRun)
    assert len(result.results) == 3
    assert len(result.request_timings_sec) == 3
    assert all(t >= 0 for t in result.request_timings_sec)
    assert result.wall_sec >= 0


@pytest.mark.asyncio
async def test_run_batch_timings_populated_for_rejected() -> None:
    pipe, _, _, _ = build(ConstantScorer(1.0))
    cfg = BatchConfig.model_validate({"n": 2, "max_concurrent": 2})
    result = await run_batch(cfg, pipe, seed=0)
    assert all(r is None for r in result.results)
    assert len(result.request_timings_sec) == 2
