"""Batch driver for the synthesis pipeline.

Loads a YAML config that defines *ranges and option lists* for every
``SynthesisRequest`` parameter, randomly samples ``n`` concrete requests
from those ranges, and runs them concurrently — bounded by a semaphore
sized to GPU capacity.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from llm import LLMClient, OpenAICompatClient
from config import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from models import CSSample
from orchestrator import SynthesisPipeline
from run_config import APIConfig, ClientConfig, CodeSwitchLinguistics
from storage.file_store import FileSampleStore

#: Backwards-compatible alias; the model now lives in run_config.
OpenAIClientConfig = ClientConfig


def make_client(config: ClientConfig) -> LLMClient:
    """Instantiate the LLM client from its YAML-embeddable config."""
    return OpenAICompatClient(**config.model_dump(exclude={"backend"}))


class BatchConfig(BaseModel):
    """Configuration for a batch synthesis run, composed from a run-config YAML.

    Pipeline settings live flat at the top level; the ``api``, ``client`` and
    ``linguistics`` groups are composed via :func:`run_config.compose_run_config`.
    """

    n: int = Field(10, ge=1, description="Total number of samples to attempt.")
    max_concurrent: int | None = Field(
        None, description="Max concurrent pipelines. None → DEFAULT_MAX_CONCURRENT."
    )
    output: str = Field(
        "out/batch",
        description="Base directory for run artifacts; the actual run lives in <output>/<run_name>.",
    )
    run_name: str = Field(
        "default",
        description="Subdirectory of `output` holding this run's samples.jsonl + profile_*.json. "
                     "Reused across invocations to resume an in-progress run.",
    )
    seed: int | None = Field(None, description="RNG seed for reproducible request sampling.")
    score_threshold: float = 7.0
    max_refinement_rounds: int = Field(3, ge=1)

    client: ClientConfig = Field(default_factory=ClientConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    linguistics: CodeSwitchLinguistics = Field(default_factory=CodeSwitchLinguistics)


#: Default in-flight pipeline cap; raise via ``max_concurrent`` in the YAML.
#: The server's continuous batching absorbs bursts — this semaphore only
#: bounds how many pipelines run concurrently client-side.
DEFAULT_MAX_CONCURRENT = 8


def sample_request(config: BatchConfig, rng: random.Random) -> SynthesisRequest:
    """Randomly draw one ``SynthesisRequest`` from the config's parameter ranges.

    ``topic`` is drawn from ``config.api.categories`` — the news-API filter
    taxonomy doubles as the topic pool.
    """
    ling = config.linguistics
    pair = rng.choice(ling.language_pairs)
    return SynthesisRequest(
        code_switching=CodeSwitchingSpec(
            type=CodeSwitchType(rng.choice(ling.cs_types)),
            function=rng.choice(ling.cs_functions),
            ratio=round(rng.uniform(ling.cs_ratio_min, ling.cs_ratio_max), 3),
        ),
        character=CharacterSetting(
            first_language=pair[0],
            second_language=pair[1],
            age=rng.randint(ling.age_min, ling.age_max),
            gender=rng.choice(ling.genders),
        ),
        basic=BasicSetting(
            perspective=rng.choice(ling.perspectives),
            tense=rng.choice(ling.tenses),
            topic=rng.choice(config.api.categories),
            conversation_type=rng.choice(ling.conversation_types),
        ),
        score_threshold=config.score_threshold,
        max_refinement_rounds=config.max_refinement_rounds,
    )


@dataclass
class BatchRun:
    """Results and timing data returned by ``run_batch``."""

    results: list[CSSample | None]
    request_timings_sec: list[float]
    wall_sec: float


def count_existing(output: str) -> int:
    """Return the number of accepted samples already written to ``output``."""
    return len(FileSampleStore(output).read_all())


async def run_batch(
    config: BatchConfig,
    pipeline: SynthesisPipeline,
    *,
    sample_fn=None,
    seed: int | None = None,
) -> BatchRun:
    """Run ``config.n`` synthesis requests concurrently and return a :class:`BatchRun`.

    Concurrency is bounded by ``config.max_concurrent`` (or ``DEFAULT_MAX_CONCURRENT``
    if unset). The server's continuous batching handles actual inference scheduling,
    so the semaphore only limits how many pipelines are in-flight at once.
    """
    _sample_fn = sample_fn if sample_fn is not None else sample_request
    n_jobs = config.max_concurrent or DEFAULT_MAX_CONCURRENT
    sem = asyncio.Semaphore(n_jobs)
    rng = random.Random(seed)
    completed = 0
    timings: list[float] = []

    async def _one(i: int) -> CSSample | None:
        nonlocal completed
        t0 = time.monotonic()
        async with sem:
            req = _sample_fn(config, rng)
            result = await pipeline.run(req)
            elapsed = time.monotonic() - t0
            timings.append(elapsed)
            completed += 1
            status = "accepted" if result is not None else "failed"
            log.info("[%d/%d] %s  topic=%r  elapsed=%.1fs", completed, config.n, status, req.topic, elapsed)
            return result

    t_start = time.monotonic()
    results = list(await asyncio.gather(*[_one(i) for i in range(config.n)]))
    return BatchRun(results=results, request_timings_sec=timings, wall_sec=time.monotonic() - t_start)
