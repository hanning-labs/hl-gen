"""Storage abstractions for accepted samples."""

from .base import SampleStore
from .file_store import FileSampleStore

__all__ = ["SampleStore", "FileSampleStore"]
