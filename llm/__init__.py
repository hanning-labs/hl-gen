"""LLM client abstractions for the synthesis framework."""

from .base import LLMClient, LLMResponse, Message
from .local import DEFAULT_MODEL as LOCAL_DEFAULT_MODEL
from .local import LocalClient
from .openai_compat import OpenAICompatClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "LocalClient",
    "LOCAL_DEFAULT_MODEL",
    "OpenAICompatClient",
]
