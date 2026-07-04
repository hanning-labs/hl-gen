"""Input parameters for the English topics synthesis pipeline."""

from __future__ import annotations

import random

from pydantic import BaseModel, Field

from batch import LocalClientConfig


class TopicsRequest(BaseModel):
    """Specification for a single English-topic synthesis run."""

    topic: str = Field(..., description="Broad topic or category to write about.")
    style: str = Field("article", description="Content style: 'news summary', 'opinion', 'explainer', 'listicle'.")
    perspective: str = Field("third-person", description="Narrative perspective.")
    tense: str = Field("present", description="Grammatical tense.")
    score_threshold: float = Field(8.0, description="S_final must meet or exceed this to accept the sample.")
    max_refinement_rounds: int = Field(3, ge=1, description="Generate→refine attempts before giving up.")


class TopicsBatchConfig(BaseModel):
    """Configuration for a batch topics synthesis run, loaded from YAML."""

    n: int = Field(10, ge=1, description="Total number of samples to attempt.")
    max_concurrent: int | None = Field(None, description="Max concurrent pipelines. None → auto-detect from GPU VRAM.")
    output: str = Field(
        "out/topics",
        description="Base directory for run artifacts; the actual run lives in <output>/<run_name>.",
    )
    run_name: str = Field(
        "default",
        description="Subdirectory of `output` holding this run's samples.jsonl + profile_*.json.",
    )
    client: LocalClientConfig = Field(default_factory=LocalClientConfig)
    seed: int | None = Field(None, description="RNG seed for reproducible request sampling.")
    score_threshold: float = 8.0
    max_refinement_rounds: int = Field(3, ge=1)

    styles: list[str] = Field(
        default_factory=lambda: ["news summary", "opinion", "explainer", "listicle"]
    )
    perspectives: list[str] = Field(default_factory=lambda: ["third-person"])
    tenses: list[str] = Field(default_factory=lambda: ["present"])
    categories: list[str] = Field(
        default_factory=lambda: [
            "general", "society", "science_technology", "politics_government",
            "economy_business_finance", "arts_culture_entertainment", "lifestyle_leisure",
            "human_interest", "sport", "crime_law_justice", "education",
            "environment", "labour", "health",
        ]
    )
    news_types: list[str] = Field(default_factory=lambda: ["news", "articles", "discussion"])
    sources: list[str] = Field(
        default_factory=lambda: ["currents", "newsapi"],
        description="News tool providers to ground generation. Subset of {'currents', 'newsapi'}.",
    )


def sample_topics_request(config: TopicsBatchConfig, rng: random.Random) -> TopicsRequest:
    """Randomly draw one TopicsRequest from the config's parameter ranges."""
    return TopicsRequest(
        topic=rng.choice(config.categories),
        style=rng.choice(config.styles),
        perspective=rng.choice(config.perspectives),
        tense=rng.choice(config.tenses),
        score_threshold=config.score_threshold,
        max_refinement_rounds=config.max_refinement_rounds,
    )
