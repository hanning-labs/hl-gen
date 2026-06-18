"""Core data models that flow through the synthesis loop.

These are the artifacts passed between agents: the candidate sample, individual
scorer verdicts, the aggregated report, refinement feedback, and the per-attempt
generation context.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .config import SynthesisRequest


class CSSample(BaseModel):
    """A candidate (or accepted) code-switched utterance."""

    text: str = Field(..., description="The code-switched sentence in the matrix language.")
    request: SynthesisRequest
    metadata: dict = Field(default_factory=dict)


class AgentScore(BaseModel):
    """One scorer's verdict on a sample (0–10)."""

    agent: str
    score: float = Field(..., ge=0.0, le=10.0)
    rationale: str = ""


class ScoreReport(BaseModel):
    """Aggregated verdict produced by the reducer (SummarizeAgent)."""

    scores: list[AgentScore] = Field(default_factory=list)
    final_score: float = Field(..., description="Weighted mean S_final across scorers.")
    passed: bool


class RefinementFeedback(BaseModel):
    """Editor (RefinerAgent) output, fed back into the next generation attempt."""

    failures: list[str] = Field(
        default_factory=list, description="Per-scorer failure explanations."
    )
    suggestions: str = ""


class GenerationContext(BaseModel):
    """Everything the generator needs for one attempt."""

    request: SynthesisRequest
    tool_context: dict = Field(
        default_factory=dict, description="External context gathered from tool providers."
    )
    feedback: RefinementFeedback | None = Field(
        None, description="Set on refinement rounds; None on the first attempt."
    )
