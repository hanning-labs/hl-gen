"""LLM client abstractions for the synthesis framework."""

from .base import LLMClient, LLMResponse, Message
from .openai_compat import OpenAICompatClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "OpenAICompatClient",
]
