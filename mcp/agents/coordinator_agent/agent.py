"""coordinator_agent — mission-level decision maker + human escalation.

Aggregates upstream analyzer results and decides the mission outcome. When a
confirmed anomaly crosses the escalation threshold and the mission's
``escalation_rules`` require a human, it **delegates** by emitting a dynamic
follow-up task for a ``human_interface`` agent — high-impact actions are never
taken without explicit human confirmation.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

AGENT_NAME = "coordinator_agent"
AGENT_VERSION = "1.0.0"


class AgentAdapter:
    """Aggregates results and coordinates human-in-the-loop escalation."""

    name = AGENT_NAME
    version = AGENT_VERSION
    capabilities = ["coordinator"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.escalation_confidence = float(config.get("escalation_confidence", 0.8))

    def handle_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        analysis = _latest_analysis(context)
        confirmed = bool(analysis.get("anomaly_confirmed"))
        confidence = float(analysis.get("confidence", 0.0))

        rules = context.get("escalation_rules") or {}
        require_human = bool(rules.get("require_human", True))
        threshold = float(rules.get("min_confidence", self.escalation_confidence))
        escalate = confirmed and confidence >= threshold and require_human

        followups: List[Dict[str, Any]] = []
        decision = "no_action"
        status = "completed"
        if escalate:
            decision = "escalate_to_human"
            status = "delegated"
            # Dynamic follow-up: require explicit human confirmation before any
            # high-impact response.
            followups.append({
                "id": f"human-confirm-{uuid.uuid4().hex[:8]}",
                "type": "human_confirm",
                "capability": "human_interface",
                "payload": {
                    "summary": "Confirmed accident — confirm emergency response?",
                    "confidence": confidence,
                    "actions": analysis.get("actions", []),
                },
            })
        elif confirmed:
            decision = "log_low_confidence_anomaly"

        return {
            "status": status,
            "result": {
                "decision": decision,
                "escalated": escalate,
                "confidence": confidence,
                "followups": followups,
            },
            "metadata": {
                "agent": AGENT_NAME, "version": AGENT_VERSION,
                "latency_ms": int((time.time() - started) * 1000),
                "request_id": uuid.uuid4().hex,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "agent": AGENT_NAME, "version": AGENT_VERSION}


def _latest_analysis(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return the most informative upstream analyzer result payload."""
    best: Dict[str, Any] = {}
    for result in (context.get("upstream") or {}).values():
        inner = (result or {}).get("result") or {}
        if "anomaly_confirmed" in inner:
            # Prefer a confirmed anomaly if any upstream reports one.
            if inner.get("anomaly_confirmed") or not best:
                best = inner
    return best
