"""Template MCP agent — copy this folder to author your own agent.

Contract (see ``dvsa_api/mcp/adapter_base.py``):

* ``handle_task(self, task: dict, context: dict) -> dict`` returning
  ``{"status": "completed"|"failed"|"in_progress"|"delegated", "result": {...},
  "metadata": {"agent": <name>, "version": <ver>, "latency_ms": int}}``.
* ``health_check(self) -> dict`` returning ``{"status": "ok"}``.
* optional ``on_start(self, config)`` / ``on_stop(self)`` lifecycle hooks.

The class **must** be named ``AgentAdapter``. It may accept a ``config`` keyword
(merged from ``manifest.json``'s ``config`` and an optional ``config.json``) or
take no arguments. Keep dependencies minimal — this template is pure stdlib.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

AGENT_NAME = "_template_agent"
AGENT_VERSION = "0.1.0"


class AgentAdapter:
    """Minimal example implementing the MCP agent contract."""

    name = AGENT_NAME
    version = AGENT_VERSION
    capabilities = ["example"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.greeting = config.get("greeting", "hello")

    def on_start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called once at registration; wire up clients/models here."""

    def on_stop(self) -> None:
        """Called on graceful shutdown; release resources here."""

    def handle_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        payload = task.get("payload") or {}
        return {
            "status": "completed",
            "result": {
                "echo": payload,
                "message": f"{self.greeting} from {AGENT_NAME}",
            },
            "metadata": {
                "agent": AGENT_NAME,
                "version": AGENT_VERSION,
                "latency_ms": int((time.time() - started) * 1000),
                "request_id": uuid.uuid4().hex,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "agent": AGENT_NAME, "version": AGENT_VERSION}
