"""LinguaMaster code-switching data synthesis framework (skeleton).

A closed-loop, multi-agent pipeline that generates, scores, and refines
code-switched text for Hanning Labs' HL-Code corpus. See ``README.md`` for the
diagram-to-code mapping. This is a skeleton: agent/LLM/tool bodies raise
``NotImplementedError`` and are implemented one-by-one in later passes; the
orchestrator loop in :mod:`orchestrator` is real.
"""

from __future__ import annotations

from .config import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from .models import (
    AgentScore,
    CSSample,
    GenerationContext,
    RefinementFeedback,
    ScoreReport,
)
from .orchestrator import SynthesisPipeline, build_default_pipeline
from .principles import LinguisticPrinciples

__all__ = [
    # config / input parameters
    "CodeSwitchType",
    "CodeSwitchingSpec",
    "CharacterSetting",
    "BasicSetting",
    "SynthesisRequest",
    "LinguisticPrinciples",
    # data models
    "CSSample",
    "AgentScore",
    "ScoreReport",
    "RefinementFeedback",
    "GenerationContext",
    # orchestration
    "SynthesisPipeline",
    "build_default_pipeline",
]
