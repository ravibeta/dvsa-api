"""Mission/session lifecycle for the MCP runtime.

The session manager plans a mission (from a named template, a template file, or
an inline spec), runs it via the :class:`~dvsa_api.mcp.executor.Executor`, stores
mission state in memory, and services control commands (``pause`` / ``resume`` /
``cancel`` / ``escalate``). It is the seam the REST API and CLI drive.

Persistence is in-memory by default; ``persist_hook`` lets a deployment mirror
mission state to Redis/DB without changing callers. Telemetry flows through the
shared :mod:`dvsa_api.mcp.metrics` sink.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapter_base import new_id, now_ms
from .errors import MissionError, MissionNotFoundError
from .executor import Executor
from .message_bus import get_bus
from .metrics import get_metrics
from .planner import Mission, plan_mission

_MISSIONS_ROOT = os.environ.get(
    "MCP_MISSIONS_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "mcp", "missions"))

_VALID_COMMANDS = ("pause", "resume", "cancel", "escalate")


@dataclass
class MissionSession:
    """In-memory record of one mission's lifecycle."""

    mission_id: str
    description: str
    status: str = "created"  # created|running|completed|partial|failed|cancelled
    created_at_ms: int = field(default_factory=now_ms)
    summary: Optional[Dict[str, Any]] = None
    mission: Optional[Mission] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    commands: List[Dict[str, Any]] = field(default_factory=list)

    def public_dict(self) -> Dict[str, Any]:
        graph = self.mission.graph.to_dict() if self.mission else {"tasks": []}
        return {
            "mission_id": self.mission_id,
            "description": self.description,
            "status": self.status,
            "created_at_ms": self.created_at_ms,
            "task_graph": graph,
            "summary": self.summary,
            "commands": self.commands,
        }


def load_template(name_or_spec: Any) -> Dict[str, Any]:
    """Resolve a mission template from an inline dict, a file path, or a name.

    A bare ``name`` resolves to ``mcp/missions/<name>.json``.
    """
    if isinstance(name_or_spec, dict):
        return name_or_spec
    if not isinstance(name_or_spec, str):
        raise MissionError("mission must be an inline object or a template name/path")
    candidates = [name_or_spec]
    if not name_or_spec.endswith(".json"):
        candidates.append(os.path.join(_MISSIONS_ROOT, f"{name_or_spec}.json"))
    else:
        candidates.append(os.path.join(_MISSIONS_ROOT, name_or_spec))
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    raise MissionError(f"mission template '{name_or_spec}' not found")


class MCPSessionManager:
    """Create, run, inspect and control missions."""

    def __init__(self, *, persist_hook: Optional[Callable[[MissionSession], None]] = None):
        self._sessions: Dict[str, MissionSession] = {}
        self._lock = threading.RLock()
        self._persist_hook = persist_hook

    # ----- lifecycle ----------------------------------------------------
    def create_mission(
        self,
        template: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        run: bool = True,
    ) -> MissionSession:
        """Plan a mission and (by default) run it to completion synchronously."""
        spec = load_template(template)
        planned = plan_mission(spec)
        session = MissionSession(
            mission_id=planned.mission_id or new_id("mission-"),
            description=planned.description,
            mission=planned,
        )
        with self._lock:
            self._sessions[session.mission_id] = session
        self._persist(session)
        if run:
            self.run_mission(session.mission_id, context or {})
        return session

    def run_mission(self, mission_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a created mission and store its summary."""
        session = self.get_mission(mission_id)
        if session.mission is None:  # pragma: no cover - defensive
            raise MissionError(f"mission '{mission_id}' has no plan")
        session.status = "running"
        self._persist(session)
        summary = Executor().run_mission(
            session.mission, context, cancel=session.cancel_event)
        session.summary = summary
        session.status = summary["status"]
        self._persist(session)
        return summary

    # ----- inspection ---------------------------------------------------
    def get_mission(self, mission_id: str) -> MissionSession:
        with self._lock:
            session = self._sessions.get(mission_id)
        if session is None:
            raise MissionNotFoundError(
                f"mission '{mission_id}' not found", details={"mission_id": mission_id})
        return session

    def list_missions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.public_dict() for s in self._sessions.values()]

    # ----- control ------------------------------------------------------
    def send_command(self, mission_id: str, command: str,
                     args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a control command to a mission."""
        if command not in _VALID_COMMANDS:
            raise MissionError(
                f"unknown command '{command}'", details={"valid": list(_VALID_COMMANDS)})
        session = self.get_mission(mission_id)
        record = {"command": command, "args": args or {}, "ts_ms": now_ms()}
        session.commands.append(record)
        if command == "cancel":
            session.cancel_event.set()
            if session.status in ("created", "running"):
                session.status = "cancelled"
        elif command == "escalate":
            get_bus().publish(
                f"mission.{mission_id}", "escalation_requested",
                {"mission_id": mission_id, "args": args or {}})
            get_metrics().event({
                "mission_id": mission_id, "kind": "escalation_requested",
                "agent": None, "task_id": None})
        # pause/resume are advisory flags recorded for the (synchronous) runner.
        self._persist(session)
        return {"mission_id": mission_id, "command": command, "accepted": True,
                "status": session.status}

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()

    def _persist(self, session: MissionSession) -> None:
        if self._persist_hook:
            try:
                self._persist_hook(session)
            except Exception:  # noqa: BLE001 - persistence must not break the run
                pass


# --- module singleton ------------------------------------------------------
_MANAGER: Optional[MCPSessionManager] = None
_MANAGER_LOCK = threading.Lock()


def get_session_manager() -> MCPSessionManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = MCPSessionManager()
        return _MANAGER


def reset_session_manager(manager: Optional[MCPSessionManager] = None) -> None:
    """Swap the module singleton (tests)."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = manager
