"""Shared agent infrastructure: role base classes and pipeline-agnostic agents."""

from .acceptance import AcceptanceAgent
from .base import (
    Agent,
    DimensionScorer,
    EditorAgent,
    GeneratorAgent,
    ReducerAgent,
    RefinerBase,
    ScorerAgent,
    SinkAgent,
)
from .summarize import SummarizeAgent

__all__ = [
    # role bases
    "Agent",
    "DimensionScorer",
    "EditorAgent",
    "GeneratorAgent",
    "ReducerAgent",
    "RefinerBase",
    "ScorerAgent",
    "SinkAgent",
    # shared concrete agents
    "AcceptanceAgent",
    "SummarizeAgent",
]
