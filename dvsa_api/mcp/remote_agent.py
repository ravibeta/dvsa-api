"""HTTP-shim proxy for ``container`` and ``remote`` MCP agents.

Container images and remote agent services implement a tiny REST contract:

* ``GET  /health``       → ``{"status": "ok"}``
* ``POST /handle_task``  → the ``{status, result, metadata}`` result

This proxy adapts that HTTP contract to the in-process :class:`AgentAdapter`
interface so the executor can treat all agent types uniformly. It is offline-safe:
with no reachable endpoint it raises :class:`AgentUnavailableError` rather than
blocking. Auth is an API key (``manifest.auth.api_key_env`` env var) or none —
**no secrets are embedded**.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, Optional

from .adapter_base import AgentAdapter, build_agent_metadata, now_ms
from .errors import AgentUnavailableError

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from .registry import AgentManifest


class RemoteAgentProxy(AgentAdapter):
    """Adapt a container/remote HTTP agent to the ``AgentAdapter`` contract."""

    def __init__(self, manifest: "AgentManifest", *, timeout_ms: int = 10000) -> None:
        self.name = manifest.name
        self.version = manifest.version
        self.capabilities = manifest.capabilities
        self.endpoint = (manifest.endpoint or "").rstrip("/")
        self.timeout_ms = timeout_ms
        auth = manifest.raw.get("auth") or {}
        key_env = auth.get("api_key_env")
        self.api_key: Optional[str] = os.environ.get(key_env) if key_env else None

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if not self.endpoint:
            raise AgentUnavailableError(
                f"remote agent '{self.name}' has no endpoint configured",
                details={"agent": self.name})
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.endpoint}{path}", data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_ms / 1000.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise AgentUnavailableError(
                f"remote agent '{self.name}' unreachable: {exc}",
                details={"agent": self.name, "endpoint": self.endpoint}) from exc

    def handle_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        started = now_ms()
        data = self._post("/handle_task", {"task": task, "context": context})
        data.setdefault("status", "completed")
        data.setdefault("result", {})
        data.setdefault("metadata", build_agent_metadata(
            agent=self.name, version=self.version, latency_ms=now_ms() - started))
        return data

    def health_check(self) -> Dict[str, Any]:
        if not self.endpoint:
            return {"status": "error", "reason": "no_endpoint"}
        try:
            req = urllib.request.Request(f"{self.endpoint}/health", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout_ms / 1000.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"status": "error", "reason": str(exc)}
