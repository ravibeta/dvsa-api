"""MCP telemetry: a counter/latency collector + a pluggable event sink.

Two small, dependency-free pieces mirroring ``dvsa_api.reasoning.metrics``:

* :class:`MCPMetrics` — thread-safe counters (``tasks_scheduled``,
  ``tasks_completed``, ``task_failures``), rolling per-agent latency (feeding
  ``avg_latency_ms``), and ``agent_health_status``.
* an *event sink* (console by default) that receives structured mission/agent
  events; swap it with :func:`set_sink` without touching call sites.

Also exposes :func:`run_with_timeout` so the executor can bound any agent call.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger("dvsa_api.mcp.metrics")


class MCPTimeout(RuntimeError):
    """Raised when an agent call exceeds its deadline."""


# --- event sink ------------------------------------------------------------
Sink = Callable[[Dict[str, Any]], None]


def console_sink(event: Dict[str, Any]) -> None:
    """Default sink: log a compact mission/agent event line."""
    logger.info(
        "mcp.event kind=%s mission=%s agent=%s task=%s status=%s",
        event.get("kind"), event.get("mission_id"), event.get("agent"),
        event.get("task_id"), event.get("status"),
    )


class MCPMetrics:
    """Thread-safe counters + rolling latency, feeding a single sink."""

    def __init__(self, *, sink: Sink = console_sink, maxlen: int = 2000) -> None:
        self._counters: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=200))
        self._health: Dict[str, str] = {}
        self._events: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._sink = sink
        self._lock = threading.Lock()

    def set_sink(self, sink: Sink) -> None:
        with self._lock:
            self._sink = sink

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe_latency(self, agent: str, latency_ms: int) -> None:
        with self._lock:
            self._latencies[agent].append(int(latency_ms))

    def set_health(self, agent: str, status: str) -> None:
        with self._lock:
            self._health[agent] = status

    def event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            sink = self._sink
        try:
            sink(event)
        except Exception:  # noqa: BLE001 - a broken sink must not fail the run
            logger.exception("mcp metrics sink raised")

    def avg_latency_ms(self, agent: Optional[str] = None) -> float:
        with self._lock:
            if agent is not None:
                samples = list(self._latencies.get(agent, ()))
            else:
                samples = [v for dq in self._latencies.values() for v in dq]
        return round(sum(samples) / len(samples), 3) if samples else 0.0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            health = dict(self._health)
            agents = list(self._latencies)
        return {
            "tasks_scheduled": counters.get("tasks_scheduled", 0),
            "tasks_completed": counters.get("tasks_completed", 0),
            "task_failures": counters.get("task_failures", 0),
            "avg_latency_ms": self.avg_latency_ms(),
            "avg_latency_ms_by_agent": {a: self.avg_latency_ms(a) for a in agents},
            "agent_health_status": health,
            "counters": counters,
        }

    def recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)[-limit:]

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()
            self._health.clear()
            self._events.clear()


# Module-level singleton used by the runtime.
_METRICS = MCPMetrics()


def get_metrics() -> MCPMetrics:
    return _METRICS


def set_sink(sink: Sink) -> None:
    _METRICS.set_sink(sink)


# --- timeout runner --------------------------------------------------------
def run_with_timeout(
    fn: Callable[[], Any], *, timeout_ms: int,
) -> Any:
    """Run ``fn`` in a daemon worker thread, raising :class:`MCPTimeout` on overrun.

    A non-positive ``timeout_ms`` runs ``fn`` inline with no guard. Python cannot
    forcibly kill a thread, so an overrunning worker is abandoned (daemon).
    """
    if timeout_ms <= 0:
        return fn()

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagate to caller thread
            error["exc"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout_ms / 1000.0)
    if worker.is_alive():
        raise MCPTimeout(f"agent call exceeded deadline_ms={timeout_ms}")
    if "exc" in error:
        raise error["exc"]
    return result.get("value")
