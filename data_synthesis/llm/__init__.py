"""LLM client abstractions for the synthesis framework."""

from .base import LLMClient, LLMResponse, Message
from .claude import DEFAULT_MODEL, ClaudeClient

__all__ = ["LLMClient", "LLMResponse", "Message", "ClaudeClient", "DEFAULT_MODEL"]
