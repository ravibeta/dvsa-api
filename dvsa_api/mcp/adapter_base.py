"""The agent adapter contract + typed task/message schemas shared MCP-wide.

An MCP *agent* — in-process shim, container, or remote service — is exposed to
the runtime through one minimal interface:

* :meth:`AgentAdapter.handle_task` — do one unit of work and return a structured
  ``{status, result, metadata}`` response.
* :meth:`AgentAdapter.health_check` — report liveness.
* optional :meth:`AgentAdapter.on_start` / :meth:`AgentAdapter.on_stop` lifecycle
  hooks.

Helpers here (:func:`new_id`, :func:`build_agent_metadata`,
:func:`ensure_result_shape`, :func:`validate_agent`) let agents emit
contract-compliant results without re-deriving the schema.
"""

from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Allowed values for a task result ``status``.
TASK_STATUSES = ("completed", "failed", "in_progress", "delegated")
RESULT_KEYS = ("status", "result", "metadata")


@dataclass
class Task:
    """One unit of work scheduled onto an agent."""

    task_id: str
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    deadline_ms: int = 15000
    priority: int = 5
    capability: Optional[str] = None
    preferred_agent: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    max_retries: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "type": self.type,
            "payload": self.payload,
            "deadline_ms": self.deadline_ms,
            "priority": self.priority,
            "capability": self.capability,
            "preferred_agent": self.preferred_agent,
            "depends_on": list(self.depends_on),
        }


@dataclass
class Message:
    """A pub/sub message on the mission message bus."""

    topic: str
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts_ms: int = 0


class AgentAdapter(abc.ABC):
    """Abstract runtime interface every MCP agent implements."""

    #: Human-readable agent name (overridden by subclasses / manifest).
    name: str = "agent"
    version: str = "0.0.0"
    #: Capabilities this agent can serve (mirrors the manifest).
    capabilities: List[str] = []

    @abc.abstractmethod
    def handle_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Do one unit of work.

        Parameters
        ----------
        task:
            ``{task_id, type, payload, deadline_ms, priority}``.
        context:
            Mission metadata ``{mission_id, session_meta, sensor_meta}``.

        Returns
        -------
        dict
            JSON-serializable ``{status, result, metadata}``.
        """

    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return ``{"status": "ok"}`` or ``{"status": "error", ...}``."""

    # Optional lifecycle hooks (no-ops by default).
    def on_start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called once when the agent is registered/started."""

    def on_stop(self) -> None:
        """Called on graceful shutdown."""


def new_id(prefix: str = "") -> str:
    """Return a short, unique id (optionally prefixed) for correlation."""
    token = uuid.uuid4().hex[:16]
    return f"{prefix}{token}" if prefix else token


def now_ms() -> int:
    """Millisecond wall-clock timestamp for latency math."""
    return int(time.time() * 1000)


def build_agent_metadata(
    *, agent: str, version: str, latency_ms: int, **extra: Any,
) -> Dict[str, Any]:
    """Assemble the ``metadata`` block required by the result contract."""
    meta: Dict[str, Any] = {"agent": agent, "version": version, "latency_ms": latency_ms}
    meta.update(extra)
    return meta


def ensure_result_shape(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate/normalise an agent result to the ``{status, result, metadata}`` shape.

    Raises ``ValueError`` (contract violation) when required keys are missing or
    ``status`` is not one of :data:`TASK_STATUSES` so bugs fail loudly in tests.
    """
    if not isinstance(result, dict):
        raise ValueError("agent result must be a dict")
    status = result.get("status")
    if status not in TASK_STATUSES:
        raise ValueError(
            f"agent result status {status!r} not in {TASK_STATUSES}")
    if "metadata" not in result:
        raise ValueError("agent result missing 'metadata'")
    normalised = dict(result)
    normalised.setdefault("result", {})
    return normalised


def validate_agent(obj: Any) -> "AgentAdapter":
    """Duck-type check that ``obj`` satisfies the agent contract.

    A foreign ``agent.py`` need not subclass :class:`AgentAdapter` as long as it
    provides callable ``handle_task`` and ``health_check`` methods.
    """
    from .errors import AgentContractError  # noqa: PLC0415 - avoid import cycle

    for method in ("handle_task", "health_check"):
        if not callable(getattr(obj, method, None)):
            raise AgentContractError(
                f"agent {obj!r} does not implement required method '{method}'")
    return obj  # type: ignore[return-value]
