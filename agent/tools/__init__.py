"""Tool registry: assembles every tool the agent can use."""
from __future__ import annotations

from langchain_core.tools import BaseTool

from agent.config import Settings
from agent.security import SecurityPolicy
from agent.tools.execution import execution_tools
from agent.tools.filesystem import filesystem_tools
from agent.tools.utility import utility_tools
from agent.tools.web import web_tools


def all_tools(settings: Settings, policy: SecurityPolicy) -> list[BaseTool]:
    """Return the full tool set for the given settings and policy."""
    tools: list[BaseTool] = []
    tools.extend(filesystem_tools(settings, policy))
    tools.extend(execution_tools(settings, policy))
    tools.extend(utility_tools())
    tools.extend(web_tools(settings, policy))
    return tools


__all__ = ["all_tools"]
