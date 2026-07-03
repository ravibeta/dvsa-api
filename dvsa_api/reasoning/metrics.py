"""Lightweight telemetry for reasoning calls + a timeout runner.

Two small, dependency-free pieces:

* :class:`MetricsCollector` — an in-memory ring buffer of per-call records with a
  pluggable *sink* (console by default). Swap the sink for StatsD/OTel/etc. by
  calling :func:`set_sink` without touching call sites.
* :func:`run_with_timeout` — run a blocking adapter ``predict`` in a worker
  thread and abort (from the caller's perspective) if it exceeds a deadline, so
  a slow/hung local adapter never blocks the request thread indefinitely.

Timeouts here are cooperative: Python cannot forcibly kill a thread, so a runaway
adapter thread is *abandoned* (daemon) rather than joined. That is acceptable for
the intended local shims; genuinely untrusted models should run out-of-process.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger("dvsa_api.reasoning.metrics")

# Abort a local adapter that runs longer than this (0 disables the guard).
REASONING_TIMEOUT_MS = int(os.environ.get("REASONING_TIMEOUT_MS", "15000"))
# Advisory ceiling used for telemetry flags; does not itself abort.
MAX_REASONING_LATENCY_MS = int(os.environ.get("MAX_REASONING_LATENCY_MS", "20000"))


class ReasoningTimeout(RuntimeError):
    """Raised when an adapter call exceeds :data:`REASONING_TIMEOUT_MS`."""


@dataclass
class CallRecord:
    """One reasoning call's telemetry (never contains secrets)."""

    request_id: str
    model_name: str
    model_version: Optional[str] = None
    latency_ms: int = 0
    tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    status: str = "ok"
    over_latency_budget: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


# --- pluggable sinks -------------------------------------------------------
Sink = Callable[[CallRecord], None]


def console_sink(record: CallRecord) -> None:
    """Default sink: emit a compact single-line log record."""
    logger.info(
        "reasoning.call request_id=%s model=%s v=%s status=%s latency_ms=%d "
        "tokens=%s effort=%s over_budget=%s",
        record.request_id, record.model_name, record.model_version,
        record.status, record.latency_ms, record.tokens,
        record.reasoning_effort, record.over_latency_budget,
    )


class MetricsCollector:
    """Thread-safe in-memory metrics buffer feeding a single sink."""

    def __init__(self, *, sink: Sink = console_sink, maxlen: int = 1000) -> None:
        self._records: Deque[CallRecord] = deque(maxlen=maxlen)
        self._sink = sink
        self._lock = threading.Lock()

    def set_sink(self, sink: Sink) -> None:
        with self._lock:
            self._sink = sink

    def record(self, record: CallRecord) -> None:
        record.over_latency_budget = (
            MAX_REASONING_LATENCY_MS > 0
            and record.latency_ms > MAX_REASONING_LATENCY_MS
        )
        with self._lock:
            self._records.append(record)
            sink = self._sink
        try:
            sink(record)
        except Exception:  # noqa: BLE001 - a broken sink must not fail the call
            logger.exception("reasoning metrics sink raised")

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._records)[-limit:]
        return [asdict(r) for r in items]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


# Module-level singleton used by the router/helpers.
_COLLECTOR = MetricsCollector()


def get_collector() -> MetricsCollector:
    return _COLLECTOR


def set_sink(sink: Sink) -> None:
    """Point the global collector at a different sink."""
    _COLLECTOR.set_sink(sink)


def record_call(record: CallRecord) -> None:
    _COLLECTOR.record(record)


# --- timeout runner --------------------------------------------------------
def run_with_timeout(
    fn: Callable[[], Any],
    *,
    timeout_ms: Optional[int] = None,
) -> Any:
    """Run ``fn`` in a worker thread, raising :class:`ReasoningTimeout` on overrun.

    ``timeout_ms`` defaults to :data:`REASONING_TIMEOUT_MS`; a non-positive value
    runs ``fn`` inline with no guard.
    """
    budget = REASONING_TIMEOUT_MS if timeout_ms is None else timeout_ms
    if budget <= 0:
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
    worker.join(budget / 1000.0)
    if worker.is_alive():
        raise ReasoningTimeout(
            f"adapter exceeded REASONING_TIMEOUT_MS={budget}ms")
    if "exc" in error:
        raise error["exc"]
    return result.get("value")
