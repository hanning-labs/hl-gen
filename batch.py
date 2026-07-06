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
from typing import Annotated, Any, Literal

log = logging.getLogger(__name__)

from pydantic import BaseModel, BeforeValidator, Field

from llm import LLMClient, LocalClient, OpenAICompatClient
from llm.local import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL as LOCAL_DEFAULT_MODEL
from llm.openai_compat import DEFAULT_BASE_URL
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

    backend: Literal["local"] = "local"
    model: str = LOCAL_DEFAULT_MODEL
    device: str | None = None
    dtype: str = "auto"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_batch_size: int = Field(1, ge=1, description="Max requests per model.generate() call. Set >1 to enable batching.")
    batch_timeout_sec: float = Field(0.02, description="Max seconds to wait for a batch to fill before firing.")
    compile_model: bool = Field(False, description="Compile the model with torch.compile for faster inference after warmup.")
    enable_thinking: bool = Field(
        False,
        description="Enable thinking mode (e.g. Qwen3+). Raise max_new_tokens (e.g. >=4096): "
                    "the <think> trace is generated before the answer and counts against the budget.",
    )


class OpenAIClientConfig(BaseModel):
    """OpenAICompatClient construction parameters, embeddable in a YAML batch config.

    Points at any OpenAI-compatible server (vLLM, SGLang, …) — local or on a
    remote VM. Batching/OOM knobs live server-side, so there are none here.
    """

    backend: Literal["openai"] = "openai"
    base_url: str = Field(DEFAULT_BASE_URL, description="Server endpoint, e.g. http://<vm>:8000/v1.")
    model: str = Field(LOCAL_DEFAULT_MODEL, description="Model id as the server knows it.")
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    api_key: str = Field("EMPTY", description="Bearer token; local vLLM/SGLang servers ignore it.")
    timeout: float = Field(600.0, description="Per-request timeout in seconds.")


def _default_backend(v: Any) -> Any:
    """Pre-existing YAMLs have no ``backend`` key; treat them as ``local``."""
    if isinstance(v, dict) and "backend" not in v:
        return {**v, "backend": "local"}
    return v


ClientConfig = Annotated[
    LocalClientConfig | OpenAIClientConfig,
    Field(discriminator="backend"),
    BeforeValidator(_default_backend),
]


def make_client(config: LocalClientConfig | OpenAIClientConfig) -> LLMClient:
    """Instantiate the LLM client selected by ``config.backend``."""
    cls = {"local": LocalClient, "openai": OpenAICompatClient}[config.backend]
    return cls(**config.model_dump(exclude={"backend"}))


class BatchConfig(BaseModel):
    """Configuration for a batch synthesis run, loaded from a YAML file."""

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
    client: ClientConfig = Field(default_factory=LocalClientConfig)
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
    conversation_types: list[str] = Field(
        default_factory=lambda: ["casual chat", "formal discussion", "debate", "storytelling"]
    )

    # currents API filters
    categories: list[str] = Field(
        default_factory=lambda: [
            "general", "society", "science_technology", "politics_government",
            "economy_business_finance", "arts_culture_entertainment", "lifestyle_leisure",
            "human_interest", "sport", "crime_law_justice", "education",
            "environment", "labour", "health", "automotive", "real_estate",
        ]
    )
    news_types: list[str] = Field(default_factory=lambda: ["news", "articles", "discussion"])


#: Default in-flight pipeline cap; raise via ``max_concurrent`` in the YAML.
#: The server's continuous batching absorbs bursts — this semaphore only
#: bounds how many pipelines run concurrently client-side.
DEFAULT_MAX_CONCURRENT = 8


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
            topic=rng.choice(config.categories),
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
