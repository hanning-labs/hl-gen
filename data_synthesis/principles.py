"""Linguistic principles that constrain generation and scoring.

These are the three constraint families from the framework's "Linguistic
Principles" block. They ride on the :class:`SynthesisRequest` and are surfaced
to the generator/scorer prompts (the actual prompt wiring is deferred to the
per-agent passes).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LinguisticPrinciples(BaseModel):
    """Constraint sets that shape and gate the synthesized utterance."""

    syntactic: list[str] = Field(
        default_factory=list,
        description="Syntactic constraints, e.g. the Equivalence / Free-Morpheme constraints.",
    )
    semantic_discourse: list[str] = Field(
        default_factory=list,
        description="Semantic and discourse-level constraints (coherence, reference, etc.).",
    )
    sociolinguistic: list[str] = Field(
        default_factory=list,
        description="Register, borrowing, and cultural-appropriateness constraints.",
    )
