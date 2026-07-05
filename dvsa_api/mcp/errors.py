"""Typed exceptions for the MCP (Multi-Agent Control Plane) runtime.

Every failure surfaced to the API maps to one of these so responses carry a
stable ``error_code`` (see :func:`error_payload`). Kept in one small module to
avoid import cycles between the registry, planner, executor and router.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MCPError(RuntimeError):
    """Base class for all MCP-runtime errors.

    ``error_code`` is a short, stable machine string; ``details`` is an optional
    JSON-serializable dict with extra context (never secrets).
    """

    error_code = "mcp_error"
    http_status = 500

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentUnavailableError(MCPError):
    """Raised when no agent can serve a task/capability."""

    error_code = "agent_unavailable"
    http_status = 503


class TaskTimeoutError(MCPError):
    """Raised when an agent task exceeds its deadline."""

    error_code = "task_timeout"
    http_status = 504


class MissionError(MCPError):
    """Raised for malformed missions or invalid mission state transitions."""

    error_code = "mission_error"
    http_status = 400


class MissionNotFoundError(MCPError):
    """Raised when a ``mission_id`` is unknown."""

    error_code = "mission_not_found"
    http_status = 404


class AgentContractError(MCPError):
    """Raised when an agent adapter violates the required contract."""

    error_code = "agent_contract_error"
    http_status = 500


class AgentNotAllowedError(MCPError):
    """Raised when an agent is blocked by the whitelist / allow-list."""

    error_code = "agent_not_allowed"
    http_status = 403


class MCPDisabledError(MCPError):
    """Raised when the MCP subsystem is called while ``ENABLE_MCP`` is off."""

    error_code = "mcp_disabled"
    http_status = 503


def error_payload(
    exc: Exception, *, fallback_actions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return the structured ``{error_code, message, details, fallback_actions}``."""
    if isinstance(exc, MCPError):
        payload = {
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        }
    else:
        payload = {"error_code": "internal_error", "message": str(exc), "details": {}}
    payload["fallback_actions"] = fallback_actions or []
    return payload
