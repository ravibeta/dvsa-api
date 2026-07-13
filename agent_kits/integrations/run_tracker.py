"""In-memory run tracking + a default dispatcher shared by the trigger adapters.

Triggers convert an external event into a :class:`RunInput`, hand it to a
*dispatcher* (which starts the work), and record status transitions in a
:class:`RunTracker` so callers can post status updates and query progress. The
default dispatcher runs the pipeline in-process (offline, deterministic); production
deployments swap in one that forwards to the cloud runner or a queue.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from ..common import RunInput, RunLogger
from ..common.pipeline import run_pipeline
from ..common.utils import deterministic_run_id

# A dispatcher starts a run for a RunInput and returns a status record.
Dispatcher = Callable[[RunInput], Dict[str, Any]]

_STATUSES = ("accepted", "running", "completed", "failed")


class RunTracker:
    """Thread-safe registry of run status for trigger-initiated runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[str, Dict[str, Any]] = {}

    def record(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        if status not in _STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._lock:
            entry = self._runs.setdefault(run_id, {"run_id": run_id, "history": []})
            entry["status"] = status
            entry["history"].append(status)
            entry.update(fields)
            return dict(entry)

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._runs.get(run_id)
            return dict(entry) if entry else None

    def all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._runs.items()}


# Process-wide default tracker (tests use their own instances).
default_tracker = RunTracker()


def make_default_dispatcher(tracker: Optional[RunTracker] = None) -> Dispatcher:
    """Return a synchronous, in-process dispatcher that records status as it runs."""
    tracker = tracker or default_tracker

    def dispatch(run_input: RunInput) -> Dict[str, Any]:
        run_id = run_input.run_id or deterministic_run_id(
            run_input.video_uri, run_input.processing_flags, prefix="trig")
        run_input.run_id = run_id
        tracker.record(run_id, "accepted", source=run_input.agent_id)
        logger = RunLogger(run_id=run_id, agent_id=run_input.agent_id)
        tracker.record(run_id, "running")
        try:
            output = run_pipeline(run_input, logger=logger)
            return tracker.record(run_id, "completed",
                                  detections=len(output.detections),
                                  summary=output.summary)
        except Exception as exc:  # noqa: BLE001 - report failure, don't crash trigger
            return tracker.record(run_id, "failed", error=str(exc))

    return dispatch


__all__ = ["RunTracker", "Dispatcher", "make_default_dispatcher", "default_tracker"]
