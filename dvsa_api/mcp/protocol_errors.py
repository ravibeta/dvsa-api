"""JSON-RPC / MCP error types for the Model-Context-Protocol server.

These map to standard JSON-RPC 2.0 error codes so responses are protocol
compliant. Kept separate from the agent control-plane ``errors.py`` because the
protocol server and the control plane are independent concerns that happen to
share the ``dvsa_api.mcp`` package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Standard JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(Exception):
    """Base error carrying a JSON-RPC ``code`` and optional ``data``."""

    code = INTERNAL_ERROR

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_error(self) -> Dict[str, Any]:
        """Return the JSON-RPC ``error`` object."""
        err: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            err["data"] = self.details
        return err


class MethodNotFoundError(JsonRpcError):
    code = METHOD_NOT_FOUND


class InvalidParamsError(JsonRpcError):
    code = INVALID_PARAMS


class InvalidRequestError(JsonRpcError):
    code = INVALID_REQUEST


class ToolNotFoundError(InvalidParamsError):
    """A ``tools/call`` referenced an unknown tool name."""


class ResourceNotFoundError(InvalidParamsError):
    """A ``resources/read`` referenced an unknown/unresolvable URI."""


class ToolExecutionError(JsonRpcError):
    """A tool handler raised while executing (surfaced as an MCP tool error)."""

    code = INTERNAL_ERROR
