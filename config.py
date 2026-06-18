"""Input Parameters for code-switching synthesis.

Mirrors the "Input Parameters" block of the LinguaMaster framework: the
code-switching spec, the speaker/character setting, and basic discourse
settings, composed into a single :class:`SynthesisRequest` that drives the
whole pipeline.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from principles import LinguisticPrinciples


class CodeSwitchType(str, Enum):
    """Structural category of the code switch."""

    INTRA_SENTENTIAL = "intra_sentential"  # switch within a clause/sentence
    INTER_SENTENTIAL = "inter_sentential"  # switch at sentence/clause boundaries
    TAG = "tag"  # tag/interjection insertion


class CodeSwitchingSpec(BaseModel):
    """The "</> Code-switching" parameters: type, function, ratio."""

    type: CodeSwitchType
    function: str = Field(
        ..., description="Discourse function of the switch, e.g. 'quotation', 'emphasis'."
    )
    ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Target fraction of embedded-language (L2) tokens.",
    )


class CharacterSetting(BaseModel):
    """The speaker persona: matrix/embedded languages and demographics."""

    first_language: str = Field(..., description="L1 / matrix language, e.g. 'Cantonese'.")
    second_language: str = Field(..., description="L2 / embedded language, e.g. 'English'.")
    age: int | str
    gender: str


class BasicSetting(BaseModel):
    """Basic discourse framing for the utterance."""

    perspective: str
    tense: str
    topic: str
    conversation_type: str


class SynthesisRequest(BaseModel):
    """Full specification for a single synthesis run (one accepted sample)."""

    code_switching: CodeSwitchingSpec
    character: CharacterSetting
    basic: BasicSetting
    principles: LinguisticPrinciples = Field(default_factory=LinguisticPrinciples)
    score_threshold: float = Field(
        8.0, description="S_final must meet or exceed this for the sample to be accepted."
    )
    max_refinement_rounds: int = Field(
        3, ge=1, description="Generate→refine attempts before the run gives up."
    )
