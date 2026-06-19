"""Tool integration (MCP-style providers) and custom-hook extension point."""

from .base import CustomHook, ToolProvider
from .gdelt import GDELTTool
from .news import GuardianTool
from .social import SocialMediaTool

__all__ = ["ToolProvider", "CustomHook", "GuardianTool", "GDELTTool", "SocialMediaTool"]
