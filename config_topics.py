"""Input parameters for the English topics synthesis pipeline."""

from __future__ import annotations

import random

from pydantic import BaseModel, Field

from run_config import APIConfig, ClientConfig, TopicsLinguistics


class TopicsRequest(BaseModel):
    """Specification for a single English-topic synthesis run."""

    topic: str = Field(..., description="Broad topic or category to write about.")
    style: str = Field("article", description="Content style: 'news summary', 'opinion', 'explainer', 'listicle'.")
    perspective: str = Field("third-person", description="Narrative perspective.")
    tense: str = Field("present", description="Grammatical tense.")
    score_threshold: float = Field(8.0, description="S_final must meet or exceed this to accept the sample.")
    max_refinement_rounds: int = Field(3, ge=1, description="Generate→refine attempts before giving up.")


class TopicsBatchConfig(BaseModel):
    """Configuration for a batch topics synthesis run, composed from a run-config YAML.

    Pipeline settings live flat at the top level; the ``api``, ``client`` and
    ``linguistics`` groups are composed via :func:`run_config.compose_run_config`.
    """

    n: int = Field(10, ge=1, description="Total number of samples to attempt.")
    max_concurrent: int | None = Field(None, description="Max concurrent pipelines. None → batch.DEFAULT_MAX_CONCURRENT.")
    output: str = Field(
        "out/topics",
        description="Base directory for run artifacts; the actual run lives in <output>/<run_name>.",
    )
    run_name: str = Field(
        "default",
        description="Subdirectory of `output` holding this run's samples.jsonl + profile_*.json.",
    )
    seed: int | None = Field(None, description="RNG seed for reproducible request sampling.")
    score_threshold: float = 8.0
    max_refinement_rounds: int = Field(3, ge=1)

    client: ClientConfig = Field(default_factory=ClientConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    linguistics: TopicsLinguistics = Field(default_factory=TopicsLinguistics)


def sample_topics_request(config: TopicsBatchConfig, rng: random.Random) -> TopicsRequest:
    """Randomly draw one TopicsRequest from the config's parameter ranges.

    ``topic`` is drawn from ``config.api.categories`` — the news-API filter
    taxonomy doubles as the topic pool.
    """
    return TopicsRequest(
        topic=rng.choice(config.api.categories),
        style=rng.choice(config.linguistics.styles),
        perspective=rng.choice(config.linguistics.perspectives),
        tense=rng.choice(config.linguistics.tenses),
        score_threshold=config.score_threshold,
        max_refinement_rounds=config.max_refinement_rounds,
    )
