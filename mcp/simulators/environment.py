"""A tiny offline environment simulator for MCP missions.

Replays sample frames/tracks from ``sample_replay/`` and injects events, then
builds the ``context`` a mission needs. :func:`simulate_mission` runs a named
mission end-to-end through the in-memory MCP runtime and returns a compact report
— no network, no external services. Used by the CLI (``mcp_cli simulate``) and
the integration tests.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(_HERE, "sample_replay")


def _load_json(name: str, default: Any) -> Any:
    path = os.path.join(SAMPLE_DIR, name)
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_sample_context(sensor_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a mission context from the bundled sample replay."""
    return {
        "tracks": _load_json("tracks.json", []),
        "frames": _load_json("frames.json", []),
        "sensor_meta": sensor_meta or {
            "gps": [51.5074, -0.1278], "altitude_m": 80, "source": "simulator"},
    }


class EnvironmentSimulator:
    """Replays sample data and lets tests inject extra events into the bus."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = _load_json("events.json", [])

    def context(self, sensor_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return load_sample_context(sensor_meta)

    def inject_event(self, kind: str, note: str = "", at_ms: int = 0) -> None:
        self.events.append({"at_ms": at_ms, "kind": kind, "note": note})


def simulate_mission(
    mission: Any = "urban_accident_response",
    *,
    sensor_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run ``mission`` against the sample replay and return a report dict.

    ``mission`` may be a template name, a path, or an inline spec.
    """
    # Imported lazily so this module stays importable without Django loaded.
    from dvsa_api.mcp import registry  # noqa: PLC0415
    from dvsa_api.mcp.session_manager import MCPSessionManager  # noqa: PLC0415

    registry.ensure_discovered()
    sim = EnvironmentSimulator()
    manager = MCPSessionManager()
    session = manager.create_mission(
        mission, context=sim.context(sensor_meta), run=True)
    summary = session.summary or {}
    return {
        "mission_id": session.mission_id,
        "status": session.status,
        "tasks": summary.get("tasks", []),
        "escalated": _was_escalated(summary),
        "injected_events": sim.events,
    }


def _was_escalated(summary: Dict[str, Any]) -> bool:
    for result in (summary.get("results") or {}).values():
        inner = (result or {}).get("result") or {}
        if inner.get("escalated"):
            return True
    return False


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    import pprint
    pprint.pprint(simulate_mission())
