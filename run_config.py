"""Composable run-configuration groups.

A run config YAML (``configs/runs/*.yaml``) holds the pipeline settings (n,
output, run_name, …) flat at the top level, plus one entry per group —
``api``, ``client``, ``linguistics``. Each group value is either the bare name
of a variant file (``api: currents`` → ``configs/api/currents.yaml``) or an
inline mapping. :func:`compose_run_config` resolves the references; the
pipeline's root model (``batch.BatchConfig`` / ``config_topics.TopicsBatchConfig``)
validates the composed result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from config import CodeSwitchType
from llm.openai_compat import DEFAULT_BASE_URL, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL

GROUPS = ("api", "client", "linguistics")


class ClientConfig(BaseModel):
    """OpenAICompatClient construction parameters (the ``client`` group).

    Points at any OpenAI-compatible server (vLLM, SGLang, …) — local or on a
    remote VM. Batching/OOM knobs live server-side, so there are none here.
    """

    backend: Literal["openai"] = "openai"
    base_url: str = Field(DEFAULT_BASE_URL, description="Server endpoint, e.g. http://<vm>:8000/v1.")
    model: str = Field(DEFAULT_MODEL, description="Model id as the server knows it.")
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    api_key: str = Field("EMPTY", description="Bearer token; local vLLM/SGLang servers ignore it.")
    timeout: float = Field(600.0, description="Per-request timeout in seconds.")


class APIConfig(BaseModel):
    """News-provider grounding parameters (the ``api`` group).

    ``categories`` doubles as the topic pool sampled into requests, so it
    must match the taxonomy of the configured ``sources``.
    """

    sources: list[str] = Field(
        default_factory=lambda: ["currents", "newsapi"],
        description="News tool providers to ground generation. Subset of {'currents', 'newsapi'}.",
    )
    categories: list[str] = Field(
        default_factory=lambda: [
            "general", "society", "science_technology", "politics_government",
            "economy_business_finance", "arts_culture_entertainment", "lifestyle_leisure",
            "human_interest", "sport", "crime_law_justice", "education",
            "environment", "labour", "health", "automotive", "real_estate",
        ]
    )
    news_types: list[str] = Field(default_factory=lambda: ["news", "articles", "discussion"])
    max_articles: int = Field(30, ge=1, description="Per-fetch article cap passed to each tool.")


class CodeSwitchLinguistics(BaseModel):
    """Sampling ranges for code-switched speech (the ``linguistics`` group)."""

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


class TopicsLinguistics(BaseModel):
    """Sampling ranges for English-topic prose (the ``linguistics`` group)."""

    styles: list[str] = Field(
        default_factory=lambda: ["news summary", "opinion", "explainer", "listicle"]
    )
    perspectives: list[str] = Field(default_factory=lambda: ["third-person"])
    tenses: list[str] = Field(default_factory=lambda: ["present"])


def compose_run_config(path: str | Path) -> dict:
    """Load a run config YAML and resolve its group references.

    For each group key, a string value ``name`` is replaced by the contents of
    ``<configs_root>/<group>/<name>.yaml`` (``configs_root`` is the run file's
    grandparent, i.e. ``configs/`` for ``configs/runs/x.yaml``); a mapping is
    kept inline as-is. Returns the composed dict, ready for ``model_validate``.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    configs_root = path.parent.parent
    for group in GROUPS:
        value = raw.get(group)
        if isinstance(value, str):
            raw[group] = yaml.safe_load((configs_root / group / f"{value}.yaml").read_text())
    return raw
