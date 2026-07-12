"""``dvsa_api.mcp`` — two independent, coexisting MCP subsystems.

1. **Model-Context-Protocol server** (Anthropic MCP) — exposes DVSA analytics as
   MCP *tools* and DVSA datasets as MCP *resources* over JSON-RPC so Claude
   Desktop / CLI or any MCP client can call DVSA natively. See :class:`MCPServer`
   / :func:`build_default_server` (modules ``server``, ``tool_registry``,
   ``resource_adapter``).

2. **Multi-Agent Control Plane** — pluggable coordinated agent workflows
   (missions/planner/executor/agents). See :class:`AgentAdapter`,
   :func:`select_agent`, :func:`plan_mission`, :class:`MCPSessionManager`.

Both are additive and opt-in; the control plane is enabled with ``ENABLE_MCP`` and
agent folders under ``mcp/agents/``, while the protocol server is launched via
``scripts/mcp_server_run.py``. Neither touches existing DVSA API signatures.
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
# Model-Context-Protocol server (independent of the control plane above).
from .server import MCPServer, build_default_server


def is_mcp_enabled() -> bool:
    """Return True when the MCP subsystem is enabled via ``ENABLE_MCP``.

    Defaults to disabled so MCP stays strictly opt-in; set ``ENABLE_MCP=true``
    (or ``1``/``yes``/``on``) to turn it on.
    """
    return os.environ.get("ENABLE_MCP", "").strip().lower() in ("1", "true", "yes", "on")


__all__ = [
    "MCPServer",
    "build_default_server",
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
