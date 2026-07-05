"""scout_agent — lightweight scout that flags candidate anomalies from tracks.

Ingests frame/track metadata and cheaply nominates regions worth deeper analysis.
Pure stdlib, deterministic, offline. When two or more object tracks are present it
raises an ``anomaly_candidate`` for a downstream analyzer to confirm.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

AGENT_NAME = "scout_agent"
AGENT_VERSION = "1.0.0"


class AgentAdapter:
    """Fast first-pass scout producing candidate anomalies."""

    name = AGENT_NAME
    version = AGENT_VERSION
    capabilities = ["scout"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.min_tracks = int(config.get("min_tracks_for_candidate", 2))

    def handle_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        payload = task.get("payload") or {}
        # Prefer live sensor data injected via mission context; fall back to the
        # task's static payload so a mission is runnable on its own.
        tracks = context.get("tracks") or payload.get("tracks") or []
        frames = context.get("frames") or payload.get("frames") or []
        candidates = []
        if len(tracks) >= self.min_tracks:
            candidates.append({
                "type": "anomaly_candidate",
                "reason": "multiple converging tracks",
                "track_ids": [t.get("id") for t in tracks],
                "confidence": 0.5,
            })
        return {
            "status": "completed",
            "result": {
                "candidates": candidates,
                "observed_tracks": len(tracks),
                "observed_frames": len(frames),
                # Hand the raw evidence forward for the analyzer.
                "tracks": tracks,
                "frames": frames,
            },
            "metadata": {
                "agent": AGENT_NAME, "version": AGENT_VERSION,
                "latency_ms": int((time.time() - started) * 1000),
                "request_id": uuid.uuid4().hex,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "agent": AGENT_NAME, "version": AGENT_VERSION}
