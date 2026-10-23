"""Agent package: a tool-using Gemini agent built on LangChain v1."""
from __future__ import annotations

from agent.builder import build_agent
from agent.config import Settings

__all__ = ["build_agent", "Settings"]
__version__ = "0.1.0"
