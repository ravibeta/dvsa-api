"""Cooperative cancellation for analytics jobs.

Same contract as the MCP executor's cancel flag (``dvsa_api/mcp/executor.py``: a
``threading.Event`` checked between waves and attempts), but backed by the job row so a
``POST /jobs/{id}/cancel/`` in the web process is seen by the Celery worker process. Stages
call :meth:`JobCancelToken.raise_if_set` at their boundaries; nothing is killed mid-stage.
"""
from __future__ import annotations

import time


class JobCancelled(Exception):
    pass


class JobCancelToken:
    def __init__(self, job_id, ttl_s: float = 0.0):
        self.job_id = job_id
        self.ttl_s = ttl_s
        self._checked_at = float("-inf")
        self._set = False

    def is_set(self) -> bool:
        if self._set:
            return True
        now = time.monotonic()
        if now - self._checked_at >= self.ttl_s:
            from .models import AnalyticsJob  # noqa: PLC0415

            self._checked_at = now
            self._set = AnalyticsJob.objects.filter(
                pk=self.job_id, status=AnalyticsJob.Status.CANCELLED).exists()
        return self._set

    def raise_if_set(self) -> None:
        if self.is_set():
            raise JobCancelled(str(self.job_id))
