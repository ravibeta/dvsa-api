"""analyzer_agent — reasoning-enabled analyzer that confirms anomalies.

Runs deeper analysis on the scout's evidence by calling a pluggable DVSA
reasoning model via ``dvsa_api.reasoning.call_model`` (default
``urban_accident_example``), returning explainable actions + a reasoning trace.
Fully offline: the reasoning model itself short-circuits to a deterministic
synthetic result, and if the reasoning layer is unavailable the agent falls back
to a local heuristic so a mission still completes.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

AGENT_NAME = "analyzer_agent"
AGENT_VERSION = "1.0.0"


class AgentAdapter:
    """Analyzer that delegates to a DVSA reasoning model when available."""

    name = AGENT_NAME
    version = AGENT_VERSION
    capabilities = ["analyzer"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.reasoning_model = config.get("reasoning_model", "urban_accident_example")

    def handle_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        tracks, frames = _gather_evidence(task, context)
        query = (task.get("payload") or {}).get(
            "query", "Confirm whether the scene contains an accident.")
        reasoning_context = {"tracks": tracks, "frames": frames, "query": query}

        actions: List[Dict[str, Any]] = []
        trace: List[str] = []
        source = "reasoning:" + self.reasoning_model
        try:
            # Reuse the pluggable reasoning layer for explainable inference.
            from dvsa_api.reasoning import call_model  # noqa: PLC0415
            out = call_model(self.reasoning_model, reasoning_context)
            actions = out.get("actions", [])
            trace = out.get("reasoning_trace", [])
        except Exception as exc:  # noqa: BLE001 - reasoning layer optional
            source = "heuristic"
            actions, trace = _heuristic(tracks)
            trace.append(f"(reasoning unavailable: {exc})")

        anomaly = next((a for a in actions if a.get("label") == "accident"), None)
        return {
            "status": "completed",
            "result": {
                "actions": actions,
                "reasoning_trace": trace,
                "anomaly_confirmed": anomaly is not None,
                "confidence": (anomaly or {}).get("confidence", 0.0),
                "source": source,
            },
            "metadata": {
                "agent": AGENT_NAME, "version": AGENT_VERSION,
                "latency_ms": int((time.time() - started) * 1000),
                "request_id": uuid.uuid4().hex,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "agent": AGENT_NAME, "version": AGENT_VERSION}


def _gather_evidence(task: Dict[str, Any], context: Dict[str, Any]):
    """Pull tracks/frames from upstream scout output, else the task payload."""
    for result in (context.get("upstream") or {}).values():
        inner = (result or {}).get("result") or {}
        if inner.get("tracks"):
            return inner.get("tracks") or [], inner.get("frames") or []
    payload = task.get("payload") or {}
    return payload.get("tracks") or [], payload.get("frames") or []


def _heuristic(tracks):
    """Offline fallback when the reasoning layer cannot be reached."""
    if len(tracks) >= 2:
        return (
            [{"type": "anomaly", "label": "accident", "confidence": 0.9}],
            ["local heuristic: two or more tracks converge -> accident"],
        )
    return (
        [{"type": "label", "label": "nominal", "confidence": 0.5}],
        ["local heuristic: insufficient tracks -> nominal"],
    )
