"""Tool integration (MCP-style providers) and custom-hook extension point."""

from .base import ToolProvider
from .currents import CurrentsTool
from .newsapi import NewsAPITool

__all__ = ["ToolProvider", "CurrentsTool", "NewsAPITool"]
