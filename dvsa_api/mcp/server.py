"""A protocol-compliant Model-Context-Protocol (MCP) server for DVSA-API.

Implements the MCP JSON-RPC 2.0 surface so Claude Desktop, the Claude CLI, or any
MCP client can call DVSA analytics as **tools** and read DVSA datasets as
**resources**:

* ``initialize`` / ``notifications/initialized`` / ``ping``
* ``tools/list`` / ``tools/call``
* ``resources/list`` / ``resources/templates/list`` / ``resources/read``

:class:`MCPServer` is transport-agnostic — :meth:`MCPServer.handle` takes a parsed
JSON-RPC request dict and returns a response dict (or ``None`` for
notifications). ``scripts/mcp_server_run.py`` wraps it in the stdio transport.

Tool handlers reuse the DVSA reasoning adapters and data layers internally, so
the DVSA core is unchanged. This protocol server is independent of the agent
*control plane* modules that also live in this package.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .protocol_errors import (
    INTERNAL_ERROR,
    InvalidRequestError,
    JsonRpcError,
    MethodNotFoundError,
)
from .resource_adapter import ResourceAdapter
from .tool_registry import ToolRegistry, build_default_registry

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "dvsa-mcp-server"
SERVER_VERSION = "1.0.0"


class MCPServer:
    """Dispatch MCP JSON-RPC requests to the DVSA tool/resource layers."""

    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        resources: Optional[ResourceAdapter] = None,
        *,
        server_name: str = SERVER_NAME,
        server_version: str = SERVER_VERSION,
    ) -> None:
        self.resources = resources or ResourceAdapter()
        self.tools = tools or build_default_registry(self.resources)
        self.server_name = server_name
        self.server_version = server_version
        self._initialized = False

    # ----- transport-agnostic entry point -------------------------------
    def handle(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle one JSON-RPC request; return a response, or None for a notification."""
        rid = request.get("id") if isinstance(request, dict) else None
        try:
            self._validate_envelope(request)
            method = request["method"]
            params = request.get("params") or {}
            is_notification = "id" not in request
            result = self._dispatch(method, params)
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except JsonRpcError as exc:
            if isinstance(request, dict) and "id" not in request:
                return None  # never respond to a notification, even on error
            return {"jsonrpc": "2.0", "id": rid, "error": exc.to_error()}
        except Exception as exc:  # noqa: BLE001 - last-resort internal error
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": INTERNAL_ERROR, "message": str(exc)}}

    def handle_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle a JSON-RPC batch, dropping notification (None) responses."""
        out = []
        for req in requests:
            resp = self.handle(req)
            if resp is not None:
                out.append(resp)
        return out

    # ----- dispatch -----------------------------------------------------
    def _dispatch(self, method: str, params: Dict[str, Any]) -> Any:
        handlers = {
            "initialize": self._initialize,
            "notifications/initialized": self._noop,
            "ping": self._ping,
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
            "resources/list": self._resources_list,
            "resources/templates/list": self._resource_templates,
            "resources/read": self._resources_read,
        }
        handler = handlers.get(method)
        if handler is None:
            raise MethodNotFoundError(f"method not found: {method}")
        return handler(params)

    def _validate_envelope(self, request: Any) -> None:
        if not isinstance(request, dict):
            raise InvalidRequestError("request must be a JSON object")
        if request.get("jsonrpc") != "2.0":
            raise InvalidRequestError("jsonrpc version must be '2.0'")
        if not isinstance(request.get("method"), str):
            raise InvalidRequestError("'method' must be a string")

    # ----- method handlers ----------------------------------------------
    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
            },
            "serverInfo": {"name": self.server_name, "version": self.server_version},
            "instructions": "DVSA-API tools for drone video analytics: anomaly "
                            "detection, reasoning, feature extraction, and dataset "
                            "resources (dvsa://frames|tracks|sensor/<id>).",
        }

    def _noop(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def _ping(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def _tools_list(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"tools": self.tools.list_tools()}

    def _tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise InvalidRequestError("tools/call requires a string 'name'")
        arguments = params.get("arguments") or {}
        try:
            structured = self.tools.call(name, arguments)
        except JsonRpcError:
            # Unknown tool / bad params are protocol errors (propagate to caller).
            raise
        except Exception as exc:  # noqa: BLE001 - defensive; registry wraps already
            return self._tool_error(str(exc))
        # MCP tool result: human-readable text content + structured content.
        return {
            "content": [{"type": "text", "text": json.dumps(structured)}],
            "structuredContent": structured,
            "isError": False,
        }

    def _tool_error(self, message: str) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}

    def _resources_list(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"resources": [r.to_mcp() for r in self.resources.list_resources()]}

    def _resource_templates(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"resourceTemplates": self.resources.resource_templates()}

    def _resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise InvalidRequestError("resources/read requires a string 'uri'")
        return self.resources.read_mcp(uri)


def build_default_server() -> MCPServer:
    """Construct an :class:`MCPServer` with the default DVSA tools + resources."""
    resources = ResourceAdapter()
    return MCPServer(build_default_registry(resources), resources)
