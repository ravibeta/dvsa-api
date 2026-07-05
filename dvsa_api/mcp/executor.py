"""Task executor — drive a mission's task graph across agents.

The executor walks the :class:`~dvsa_api.mcp.planner.TaskGraph` in dependency
order, assigning each ready task to an agent (a task's ``preferred_agent`` or one
chosen by :func:`~dvsa_api.mcp.registry.select_agent` for its capability). It:

* enforces a **per-task timeout** (``deadline_ms``) via a bounded runner,
* **retries** failed/timed-out tasks per the mission policy (with backoff) and
  then tries the manifest/mission **fallback agents**,
* runs a dependency *wave* **concurrently** up to ``MCP_MAX_CONCURRENCY``,
* schedules **dynamic follow-up tasks** an agent may emit (``result.followups``),
* publishes lifecycle **events** to the message bus and records **metrics**,
* supports cooperative **cancellation** via a shared flag.

It is synchronous (runs a mission to completion) and deterministic for tests.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .adapter_base import Task, now_ms
from .errors import AgentUnavailableError, TaskTimeoutError
from .message_bus import InMemoryBus, get_bus
from .metrics import MCPTimeout, get_metrics, run_with_timeout
from .planner import Mission, TaskNode
from . import registry

DEFAULT_MAX_CONCURRENCY = int(os.environ.get("MCP_MAX_CONCURRENCY", "4"))


class Executor:
    """Runs a planned :class:`Mission` to completion."""

    def __init__(
        self,
        *,
        bus: Optional[InMemoryBus] = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self.bus = bus or get_bus()
        self.metrics = get_metrics()
        self.max_concurrency = max(1, max_concurrency)

    # ----- public API ---------------------------------------------------
    def run_mission(
        self,
        mission: Mission,
        context: Dict[str, Any],
        *,
        cancel: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Execute ``mission`` and return a summary dict."""
        cancel = cancel or threading.Event()
        started = now_ms()
        deadline = started + mission.mission_timeout_ms
        ctx = dict(context)
        ctx.setdefault("mission_id", mission.mission_id)
        ctx["escalation_rules"] = mission.escalation_rules

        self._emit(mission.mission_id, "mission_started", None, None, {
            "description": mission.description})

        while not mission.graph.is_done():
            if cancel.is_set():
                self._emit(mission.mission_id, "mission_cancelled", None, None, {})
                break
            if now_ms() > deadline:
                self._fail_remaining(mission, "mission timeout")
                self._emit(mission.mission_id, "mission_timeout", None, None, {})
                break
            wave = mission.graph.ready_tasks()
            if not wave:
                # Nothing runnable but not done → unmet deps blocked by failures.
                self._fail_remaining(mission, "blocked by failed dependency")
                break
            self._run_wave(mission, wave, ctx, cancel)

        status = self._mission_status(mission, cancel)
        summary = {
            "mission_id": mission.mission_id,
            "status": status,
            "latency_ms": now_ms() - started,
            "tasks": mission.graph.to_dict()["tasks"],
            "results": {n.task.task_id: n.result for n in mission.graph.nodes()
                        if n.result is not None},
        }
        self._emit(mission.mission_id, "mission_finished", None, None,
                   {"status": status})
        return summary

    # ----- wave execution ----------------------------------------------
    def _run_wave(
        self, mission: Mission, wave: List[TaskNode],
        ctx: Dict[str, Any], cancel: threading.Event,
    ) -> None:
        for node in wave:
            node.status = "running"
        workers = min(self.max_concurrency, len(wave))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._run_task, mission, n, ctx, cancel): n
                       for n in wave}
            for future in futures:
                future.result()  # exceptions are handled inside _run_task

    def _run_task(
        self, mission: Mission, node: TaskNode,
        ctx: Dict[str, Any], cancel: threading.Event,
    ) -> None:
        """Run one task with retries + fallback; mutate ``node`` in place."""
        task = node.task
        self.metrics.incr("tasks_scheduled")
        agents = self._agent_candidates(mission, task)
        last_error = "no agent available"
        # Make completed upstream results available to this task's agent.
        task_ctx = dict(ctx)
        task_ctx["upstream"] = {
            dep: mission.graph.node(dep).result
            for dep in task.depends_on
            if mission.graph.node(dep).result is not None}

        for agent_name in agents:
            attempt = 0
            while attempt <= task.max_retries:
                if cancel.is_set():
                    node.status = "failed"
                    node.error = "cancelled"
                    return
                node.attempts += 1
                attempt += 1
                node.assigned_agent = agent_name
                try:
                    result = self._invoke(agent_name, task, task_ctx)
                    node.result = result
                    node.status = result.get("status", "completed")
                    self.metrics.incr("tasks_completed")
                    self._emit(mission.mission_id, "task_completed",
                               agent_name, task.task_id,
                               {"status": node.status})
                    self._spawn_followups(mission, node, result)
                    return
                except (TaskTimeoutError, MCPTimeout) as exc:
                    last_error = f"timeout: {exc}"
                    self._emit(mission.mission_id, "task_timeout",
                               agent_name, task.task_id, {})
                except AgentUnavailableError as exc:
                    last_error = f"unavailable: {exc}"
                    break  # try the next fallback agent, don't retry this one
                except Exception as exc:  # noqa: BLE001 - agent raised
                    last_error = f"error: {exc}"
                if attempt <= task.max_retries:
                    time.sleep(mission.backoff_ms / 1000.0)

        node.status = "failed"
        node.error = last_error
        self.metrics.incr("task_failures")
        self._emit(mission.mission_id, "task_failed", node.assigned_agent,
                   task.task_id, {"error": last_error})

    def _invoke(
        self, agent_name: str, task: Task, ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call an agent's ``handle_task`` under the task deadline."""
        agent = registry.get_agent(agent_name)
        started = now_ms()
        try:
            result = run_with_timeout(
                lambda: agent.handle_task(task.to_dict(), ctx),
                timeout_ms=task.deadline_ms)
        except MCPTimeout as exc:
            raise TaskTimeoutError(
                f"agent '{agent_name}' timed out on task '{task.task_id}'",
                details={"deadline_ms": task.deadline_ms}) from exc
        latency = now_ms() - started
        self.metrics.observe_latency(agent_name, latency)
        if not isinstance(result, dict):
            raise ValueError(f"agent '{agent_name}' returned non-dict result")
        result.setdefault("status", "completed")
        result.setdefault("result", {})
        result.setdefault("metadata", {})
        result["metadata"].setdefault("agent", agent_name)
        result["metadata"].setdefault("latency_ms", latency)
        return result

    # ----- helpers ------------------------------------------------------
    def _agent_candidates(self, mission: Mission, task: Task) -> List[str]:
        """Ordered agent names to try: preferred/selected, then fallbacks."""
        candidates: List[str] = []
        if task.preferred_agent:
            candidates.append(task.preferred_agent)
        elif task.capability:
            try:
                candidates.append(
                    registry.select_agent(task.capability, mission.selection_policy))
            except AgentUnavailableError:
                pass
        # Manifest-declared fallbacks for the first candidate.
        if candidates:
            manifest = registry.get_manifest(candidates[0])
            if manifest:
                candidates.extend(manifest.fallback_agents)
        # Mission-level fallbacks (by capability) as a last resort.
        if task.capability:
            for m in registry.agents_for_capability(task.capability):
                if m.name not in candidates:
                    candidates.append(m.name)
        # De-dupe, preserve order.
        seen: set = set()
        ordered = [c for c in candidates if not (c in seen or seen.add(c))]
        return ordered

    def _spawn_followups(
        self, mission: Mission, node: TaskNode, result: Dict[str, Any],
    ) -> None:
        """Add any dynamic follow-up tasks the agent emitted."""
        followups = (result.get("result") or {}).get("followups") or result.get(
            "followups") or []
        for spec in followups:
            tid = spec.get("id") or f"{node.task.task_id}-f{node.attempts}"
            try:
                new_node = mission.graph.add_task(Task(
                    task_id=tid,
                    type=spec.get("type", "followup"),
                    payload=spec.get("payload") or {},
                    deadline_ms=int(spec.get("deadline_ms", mission.task_timeout_ms)),
                    priority=int(spec.get("priority", 5)),
                    capability=spec.get("capability"),
                    preferred_agent=spec.get("agent"),
                    depends_on=list(spec.get("depends_on") or []),
                    max_retries=int(spec.get("max_retries", mission.max_retries)),
                ))
                self._emit(mission.mission_id, "task_spawned", None,
                           new_node.task.task_id, {"parent": node.task.task_id})
            except Exception:  # noqa: BLE001 - duplicate id etc.; ignore
                continue

    def _fail_remaining(self, mission: Mission, reason: str) -> None:
        for node in mission.graph.nodes():
            if node.status in ("pending", "ready", "running"):
                node.status = "failed"
                node.error = reason
                self.metrics.incr("task_failures")

    def _mission_status(self, mission: Mission, cancel: threading.Event) -> str:
        if cancel.is_set():
            return "cancelled"
        if mission.graph.has_failures():
            done = any(n.status in ("completed", "delegated")
                       for n in mission.graph.nodes())
            return "partial" if done else "failed"
        return "completed"

    def _emit(
        self, mission_id: str, kind: str, agent: Optional[str],
        task_id: Optional[str], extra: Dict[str, Any],
    ) -> None:
        event = {"mission_id": mission_id, "kind": kind, "agent": agent,
                 "task_id": task_id, "status": extra.get("status"), **extra}
        self.metrics.event(event)
        self.bus.publish(f"mission.{mission_id}", kind, event)
