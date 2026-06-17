"""Tool integration (MCP-style providers) and custom-hook extension point."""

from .base import CustomHook, ToolProvider
from .news import NewsAPITool
from .social import SocialMediaTool

__all__ = ["ToolProvider", "CustomHook", "NewsAPITool", "SocialMediaTool"]
