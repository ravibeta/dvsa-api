"""MCP (Multi-Agent Control Plane) — pluggable coordinated agent workflows.

Public surface
--------------
* :class:`AgentAdapter` — the agent contract every agent implements.
* registry helpers — :func:`get_agent`, :func:`list_agents`, :func:`select_agent`,
  :func:`discover_agents`.
* planning/execution — :func:`plan_mission`, :class:`Executor`.
* :class:`MCPSessionManager` / :func:`get_session_manager` — mission lifecycle.

MCP is **additive and opt-in**: enable it by setting ``ENABLE_MCP=true`` and
dropping agent folders under ``mcp/agents/``. Nothing here touches existing DVSA
APIs, and the default in-memory message bus needs no external services.
"""

from __future__ import annotations

import os

from .adapter_base import AgentAdapter
from .errors import (
    AgentContractError,
    AgentUnavailableError,
    MCPError,
    MissionError,
    MissionNotFoundError,
    TaskTimeoutError,
)
from .executor import Executor
from .planner import Mission, plan_mission
from .registry import (
    AgentManifest,
    discover_agents,
    ensure_discovered,
    get_agent,
    get_manifest,
    list_agents,
    list_manifests,
    register,
    register_instance,
    select_agent,
)
from .session_manager import (
    MCPSessionManager,
    MissionSession,
    get_session_manager,
    reset_session_manager,
)


def is_mcp_enabled() -> bool:
    """Return True when the MCP subsystem is enabled via ``ENABLE_MCP``.

    Defaults to disabled so MCP stays strictly opt-in; set ``ENABLE_MCP=true``
    (or ``1``/``yes``/``on``) to turn it on.
    """
    return os.environ.get("ENABLE_MCP", "").strip().lower() in ("1", "true", "yes", "on")


__all__ = [
    "AgentAdapter",
    "AgentManifest",
    "Executor",
    "Mission",
    "MCPSessionManager",
    "MissionSession",
    "MCPError",
    "AgentContractError",
    "AgentUnavailableError",
    "MissionError",
    "MissionNotFoundError",
    "TaskTimeoutError",
    "plan_mission",
    "discover_agents",
    "ensure_discovered",
    "get_agent",
    "get_manifest",
    "list_agents",
    "list_manifests",
    "register",
    "register_instance",
    "select_agent",
    "get_session_manager",
    "reset_session_manager",
    "is_mcp_enabled",
]
