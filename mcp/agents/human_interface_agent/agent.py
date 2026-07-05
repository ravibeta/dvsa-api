"""human_interface_agent — human-in-the-loop confirmation stub.

Represents the point where a human confirms or annotates a high-impact decision.
In a real deployment ``handle_task`` would post to a webhook / UI and return
``status: "in_progress"`` until a person responds. For deterministic, offline
tests and simulations it auto-confirms when ``MCP_HUMAN_AUTO_CONFIRM`` is set
(the default), returning an ``approved`` decision.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

AGENT_NAME = "human_interface_agent"
AGENT_VERSION = "1.0.0"


def _auto_confirm_default() -> bool:
    # Default ON so missions complete offline; set MCP_HUMAN_AUTO_CONFIRM=0 to
    # require a real out-of-band response (agent then reports in_progress).
    return os.environ.get("MCP_HUMAN_AUTO_CONFIRM", "1").strip().lower() in (
        "1", "true", "yes", "on")


class AgentAdapter:
    """Mock human confirmation gate for high-impact actions."""

    name = AGENT_NAME
    version = AGENT_VERSION
    capabilities = ["human_interface"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.webhook_url = config.get("webhook_url")

    def handle_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        payload = task.get("payload") or {}
        auto = _auto_confirm_default()
        if auto:
            status, decision, confirmed = "completed", "approved", True
        else:
            # Awaiting a real human response (webhook/UI) — not yet resolved.
            status, decision, confirmed = "in_progress", "pending", False
        return {
            "status": status,
            "result": {
                "confirmed": confirmed,
                "decision": decision,
                "prompt": payload.get("summary", "Confirm action?"),
                "channel": "webhook" if self.webhook_url else "mock",
            },
            "metadata": {
                "agent": AGENT_NAME, "version": AGENT_VERSION,
                "latency_ms": int((time.time() - started) * 1000),
                "request_id": uuid.uuid4().hex,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "agent": AGENT_NAME, "version": AGENT_VERSION}
