"""Linguistic principles that constrain generation and scoring.

These are the three constraint families from the framework's "Linguistic
Principles" block. They ride on the :class:`SynthesisRequest` and are surfaced
to the generator/scorer prompts (the actual prompt wiring is deferred to the
per-agent passes).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


def _default_syntactic() -> list[str]:
    return [
        "Equivalence Constraint: switch only at points where the surface syntax of both "
        "languages aligns — do not reorder constituents to force a switch site.",
        "Free Morpheme Constraint: never attach a bound morpheme from one language to a "
        "stem from another (e.g. avoid *去-ed where '去' is Cantonese and '-ed' is English).",
        "Functional-Head Constraint: do not switch between a functional head "
        "(complementizer, determiner, or inflectional element) and its complement — "
        "only switch at spans where both functional projections are compatible.",
    ]


def _default_semantic_discourse() -> list[str]:
    return [
        "Referential coherence: pronouns and definite NPs must resolve to the same "
        "referent across a switch boundary.",
        "Topic continuity: the switched segment should advance or reinforce the current "
        "discourse topic, not introduce an unrelated one.",
        "Illocutionary clarity: the speech act (question, request, statement) must remain "
        "unambiguous even when surface markers shift language.",
        "Contextual motivation (Auer 1998): every switch must be recoverable from "
        "discourse context — topic shift, participant design, or domain — so that a "
        "bilingual listener can identify why the switch occurred rather than experiencing "
        "it as arbitrary.",
    ]


def _default_sociolinguistic() -> list[str]:
    return [
        "Use community-established loanwords and calques rather than ad-hoc translations "
        "(e.g. Cantonese '士多啤梨' over '草莓' for 'strawberry'; keep 'taco' in Spanish "
        "rather than forcing a translation).",
        "Match register across the switch: do not combine a formal L1 register with "
        "casual L2 slang unless the contrast is the intended rhetorical effect.",
        "Respect domain conventions: technical terms, brand names, and emotional "
        "interjections are high-frequency switch triggers; respect what sounds natural "
        "to the target bilingual community.",
        "Avoid offensive or contextually inappropriate usage: do not produce switches "
        "that would be considered rude, taboo, or culturally tone-deaf in the local "
        "bilingual community, even if the individual words are benign in isolation.",
    ]


class LinguisticPrinciples(BaseModel):
    """Constraint sets that shape and gate the synthesized utterance."""

    syntactic: list[str] = Field(
        default_factory=_default_syntactic,
        description="Syntactic constraints, e.g. the Equivalence / Free-Morpheme constraints.",
    )
    semantic_discourse: list[str] = Field(
        default_factory=_default_semantic_discourse,
        description="Semantic and discourse-level constraints (coherence, reference, etc.).",
    )
    sociolinguistic: list[str] = Field(
        default_factory=_default_sociolinguistic,
        description="Register, borrowing, and cultural-appropriateness constraints.",
    )
