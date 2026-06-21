"""The eight LinguaMaster agents and their role base classes."""

from .acceptance import AcceptanceAgent
from .article_selector import ArticleSelectorAgent
from .base import (
    Agent,
    EditorAgent,
    GeneratorAgent,
    ReducerAgent,
    ScorerAgent,
    SinkAgent,
)
from .generation import GenerationAgent
from .refiner import RefinerAgent
from .scorers import (
    CSRatioAgent,
    FluencyAgent,
    NaturalnessAgent,
    SocialCultureAgent,
)
from .summarize import SummarizeAgent

__all__ = [
    # role bases
    "Agent",
    # pre-generation
    "ArticleSelectorAgent",
    "GeneratorAgent",
    "ScorerAgent",
    "ReducerAgent",
    "EditorAgent",
    "SinkAgent",
    # concrete agents
    "GenerationAgent",
    "FluencyAgent",
    "NaturalnessAgent",
    "CSRatioAgent",
    "SocialCultureAgent",
    "SummarizeAgent",
    "RefinerAgent",
    "AcceptanceAgent",
]
