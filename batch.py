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
from typing import Any

log = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from llm.local import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL as LOCAL_DEFAULT_MODEL
from config import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from models import CSSample
from orchestrator import SynthesisPipeline
from storage.file_store import FileSampleStore


class LocalClientConfig(BaseModel):
    """LocalClient construction parameters, embeddable in a YAML batch config."""

    model: str = LOCAL_DEFAULT_MODEL
    device: str | None = None
    dtype: str = "auto"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_batch_size: int = Field(1, ge=1, description="Max requests per model.generate() call. Set >1 to enable batching.")
    batch_timeout_sec: float = Field(0.02, description="Max seconds to wait for a batch to fill before firing.")
    compile_model: bool = Field(False, description="Compile the model with torch.compile for faster inference after warmup.")


class BatchConfig(BaseModel):
    """Configuration for a batch synthesis run, loaded from a YAML file."""

    n: int = Field(10, ge=1, description="Total number of samples to attempt.")
    max_concurrent: int | None = Field(
        None, description="Max concurrent pipelines. None → auto-detect from GPU VRAM."
    )
    output: str = "out/batch_samples.jsonl"
    client: LocalClientConfig = Field(default_factory=LocalClientConfig)
    seed: int | None = Field(None, description="RNG seed for reproducible request sampling.")
    score_threshold: float = 7.0
    max_refinement_rounds: int = Field(3, ge=1)

    # code_switching ranges
    cs_types: list[str] = Field(
        default_factory=lambda: [t.value for t in CodeSwitchType],
        description="CodeSwitchType values to draw from.",
    )
    cs_functions: list[str] = Field(
        default_factory=lambda: ["expressive", "quotation", "emphasis", "clarification"]
    )
    cs_ratio_min: float = Field(0.2, ge=0.0, le=1.0)
    cs_ratio_max: float = Field(0.5, ge=0.0, le=1.0)

    # character ranges
    language_pairs: list[list[str]] = Field(
        default_factory=lambda: [["Cantonese", "English"]],
        description="List of [L1, L2] pairs to draw from.",
    )
    age_min: int = Field(18, ge=0)
    age_max: int = Field(60, ge=0)
    genders: list[str] = Field(default_factory=lambda: ["male", "female", "non-binary"])

    # basic ranges
    perspectives: list[str] = Field(default_factory=lambda: ["first-person", "third-person"])
    tenses: list[str] = Field(default_factory=lambda: ["past", "present", "future"])
    topics: list[str] = Field(
        default_factory=lambda: ["technology", "movies", "food", "work", "family", "travel"]
    )
    conversation_types: list[str] = Field(
        default_factory=lambda: ["casual chat", "formal discussion", "debate", "storytelling"]
    )


def default_max_concurrent() -> int:
    """Estimate max concurrent pipelines from available GPU VRAM (~1 per 8 GB)."""
    try:
        import torch
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return max(1, int(mem_gb // 8))
    except ImportError:
        pass
    return 1


def sample_request(config: BatchConfig, rng: random.Random) -> SynthesisRequest:
    """Randomly draw one ``SynthesisRequest`` from the config's parameter ranges."""
    pair = rng.choice(config.language_pairs)
    return SynthesisRequest(
        code_switching=CodeSwitchingSpec(
            type=CodeSwitchType(rng.choice(config.cs_types)),
            function=rng.choice(config.cs_functions),
            ratio=round(rng.uniform(config.cs_ratio_min, config.cs_ratio_max), 3),
        ),
        character=CharacterSetting(
            first_language=pair[0],
            second_language=pair[1],
            age=rng.randint(config.age_min, config.age_max),
            gender=rng.choice(config.genders),
        ),
        basic=BasicSetting(
            perspective=rng.choice(config.perspectives),
            tense=rng.choice(config.tenses),
            topic=rng.choice(config.topics),
            conversation_type=rng.choice(config.conversation_types),
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
    seed: int | None = None,
) -> BatchRun:
    """Run ``config.n`` synthesis requests concurrently and return a :class:`BatchRun`.

    Concurrency is bounded by ``config.max_concurrent`` (or ``default_max_concurrent()``
    if unset). ``LocalClient._lock`` serializes actual GPU inference, so the semaphore
    only limits how many pipelines are in-flight at once.
    """
    n_jobs = config.max_concurrent or default_max_concurrent()
    sem = asyncio.Semaphore(n_jobs)
    rng = random.Random(seed)
    completed = 0
    timings: list[float] = []

    async def _one(i: int) -> CSSample | None:
        nonlocal completed
        t0 = time.monotonic()
        async with sem:
            req = sample_request(config, rng)
            result = await pipeline.run(req)
            elapsed = time.monotonic() - t0
            timings.append(elapsed)
            completed += 1
            status = "accepted" if result is not None else "failed"
            log.info("[%d/%d] %s  topic=%r  L1=%s  elapsed=%.1fs", completed, config.n, status, req.basic.topic, req.character.first_language, elapsed)
            return result

    t_start = time.monotonic()
    results = list(await asyncio.gather(*[_one(i) for i in range(config.n)]))
    return BatchRun(results=results, request_timings_sec=timings, wall_sec=time.monotonic() - t_start)
