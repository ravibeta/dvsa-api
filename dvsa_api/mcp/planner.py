"""Mission planner — turn a mission template into an executable task DAG.

A *mission template* (JSON) describes ``tasks`` (a graph with dependencies), the
``agents``/capabilities they require, ``timeouts``, ``retry_policy`` and
``escalation_rules``. :func:`plan_mission` validates it (including cycle
detection) and produces a :class:`Mission` with a :class:`TaskGraph` the executor
can drive. Agents may emit follow-up tasks at run time, which the executor adds
back onto the graph via :meth:`TaskGraph.add_task` (dynamic task generation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adapter_base import Task, new_id
from .errors import MissionError

# Sensible safe defaults so a minimal template is still runnable.
DEFAULT_TASK_TIMEOUT_MS = 15000
DEFAULT_MISSION_TIMEOUT_MS = 120000
DEFAULT_MAX_RETRIES = 1
DEFAULT_BACKOFF_MS = 100


@dataclass
class TaskNode:
    """A task plus its mutable execution state within a mission."""

    task: Task
    status: str = "pending"  # pending|ready|running|completed|failed|delegated
    attempts: int = 0
    assigned_agent: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "type": self.task.type,
            "capability": self.task.capability,
            "preferred_agent": self.task.preferred_agent,
            "depends_on": list(self.task.depends_on),
            "status": self.status,
            "attempts": self.attempts,
            "assigned_agent": self.assigned_agent,
            "error": self.error,
        }


class TaskGraph:
    """A DAG of :class:`TaskNode` with dependency-aware scheduling helpers."""

    def __init__(self) -> None:
        self._nodes: Dict[str, TaskNode] = {}

    # ----- construction -------------------------------------------------
    def add_task(self, task: Task) -> TaskNode:
        """Add a task (used at plan time and for dynamic follow-ups)."""
        if task.task_id in self._nodes:
            raise MissionError(f"duplicate task_id '{task.task_id}'")
        node = TaskNode(task=task)
        self._nodes[task.task_id] = node
        return node

    def validate(self) -> None:
        """Ensure all dependencies exist and the graph is acyclic."""
        for node in self._nodes.values():
            for dep in node.task.depends_on:
                if dep not in self._nodes:
                    raise MissionError(
                        f"task '{node.task.task_id}' depends on unknown '{dep}'")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        # Kahn's algorithm — if some nodes never reach in-degree 0, there's a cycle.
        indeg = {tid: 0 for tid in self._nodes}
        for node in self._nodes.values():
            for dep in node.task.depends_on:
                indeg[node.task.task_id] += 1
        queue = [tid for tid, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            tid = queue.pop()
            seen += 1
            for other in self._nodes.values():
                if tid in other.task.depends_on:
                    indeg[other.task.task_id] -= 1
                    if indeg[other.task.task_id] == 0:
                        queue.append(other.task.task_id)
        if seen != len(self._nodes):
            raise MissionError("mission task graph contains a cycle")

    # ----- scheduling ---------------------------------------------------
    def ready_tasks(self) -> List[TaskNode]:
        """Pending tasks whose dependencies have all completed."""
        ready: List[TaskNode] = []
        for node in self._nodes.values():
            if node.status not in ("pending", "ready"):
                continue
            if all(self._is_done(dep) for dep in node.task.depends_on):
                node.status = "ready"
                ready.append(node)
        # Highest priority first (lower ``priority`` number == more urgent).
        ready.sort(key=lambda n: n.task.priority)
        return ready

    def _is_done(self, task_id: str) -> bool:
        node = self._nodes.get(task_id)
        return node is not None and node.status in ("completed", "delegated")

    def is_done(self) -> bool:
        """True when no task can make further progress."""
        return all(
            n.status in ("completed", "failed", "delegated")
            for n in self._nodes.values())

    def has_failures(self) -> bool:
        return any(n.status == "failed" for n in self._nodes.values())

    # ----- access -------------------------------------------------------
    def node(self, task_id: str) -> TaskNode:
        if task_id not in self._nodes:
            raise MissionError(f"unknown task '{task_id}'")
        return self._nodes[task_id]

    def nodes(self) -> List[TaskNode]:
        return list(self._nodes.values())

    def to_dict(self) -> Dict[str, Any]:
        return {"tasks": [n.to_dict() for n in self._nodes.values()]}


@dataclass
class Mission:
    """A planned mission: metadata + a ready-to-execute task graph."""

    mission_id: str
    description: str
    graph: TaskGraph
    task_timeout_ms: int = DEFAULT_TASK_TIMEOUT_MS
    mission_timeout_ms: int = DEFAULT_MISSION_TIMEOUT_MS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_ms: int = DEFAULT_BACKOFF_MS
    selection_policy: str = "round_robin"
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def plan_mission(template: Dict[str, Any]) -> Mission:
    """Validate a mission ``template`` and build its :class:`Mission`.

    Raises :class:`MissionError` for a missing ``tasks`` list, duplicate task ids,
    unknown dependencies, or a cyclic graph.
    """
    if not isinstance(template, dict):
        raise MissionError("mission template must be a JSON object")
    tasks = template.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise MissionError("mission template requires a non-empty 'tasks' list")

    timeouts = template.get("timeouts") or {}
    retry = template.get("retry_policy") or {}
    task_timeout = int(timeouts.get("task_ms", DEFAULT_TASK_TIMEOUT_MS))
    max_retries = int(retry.get("max_retries", DEFAULT_MAX_RETRIES))

    graph = TaskGraph()
    for spec in tasks:
        if "id" not in spec:
            raise MissionError("each task requires an 'id'")
        if not spec.get("capability") and not spec.get("agent"):
            raise MissionError(
                f"task '{spec['id']}' needs a 'capability' or explicit 'agent'")
        graph.add_task(Task(
            task_id=str(spec["id"]),
            type=spec.get("type", spec["id"]),
            payload=spec.get("payload") or {},
            deadline_ms=int(spec.get("deadline_ms", task_timeout)),
            priority=int(spec.get("priority", 5)),
            capability=spec.get("capability"),
            preferred_agent=spec.get("agent"),
            depends_on=list(spec.get("depends_on") or []),
            max_retries=int(spec.get("max_retries", max_retries)),
        ))
    graph.validate()

    return Mission(
        mission_id=str(template.get("mission_id") or new_id("mission-")),
        description=str(template.get("description", "")),
        graph=graph,
        task_timeout_ms=task_timeout,
        mission_timeout_ms=int(timeouts.get("mission_ms", DEFAULT_MISSION_TIMEOUT_MS)),
        max_retries=max_retries,
        backoff_ms=int(retry.get("backoff_ms", DEFAULT_BACKOFF_MS)),
        selection_policy=str(template.get("selection_policy", "round_robin")),
        escalation_rules=template.get("escalation_rules") or {},
        triggers=list(template.get("triggers") or []),
        raw=template,
    )
