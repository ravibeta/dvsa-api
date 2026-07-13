"""Structured, observability-friendly logging for agent kits.

Every log line is a single JSON object carrying the observability fields the kits
must emit — ``run_id``, ``agent_id``, ``model_version``, ``trace_id`` — plus a
timestamp, level and message. JSON keeps logs machine-parseable for dashboards
while remaining greppable in a terminal.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, Optional

_LOGGER_NAME = "agent_kits"
_CONTEXT_KEYS = ("run_id", "agent_id", "model_version", "trace_id")


class JsonFormatter(logging.Formatter):
    """Render log records (and their kit context) as one-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _CONTEXT_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure and return the shared ``agent_kits`` logger (idempotent)."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        # Logs go to stderr so a kit's stdout carries only its JSON result/data.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel((level or os.environ.get("AGENT_KITS_LOG_LEVEL", "INFO")).upper())
    return logger


def new_trace_id() -> str:
    """Return a fresh trace id for correlating a single run's log lines."""
    return uuid.uuid4().hex


class RunLogger:
    """Logger bound to a run's observability context.

    Usage::

        log = RunLogger(run_id="run_abc", agent_id="scout", model_version="v1")
        log.info("started", frames=120)
    """

    def __init__(
        self,
        *,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model_version: Optional[str] = None,
        trace_id: Optional[str] = None,
        level: Optional[str] = None,
    ) -> None:
        self._logger = configure_logging(level)
        self.context = {
            "run_id": run_id,
            "agent_id": agent_id,
            "model_version": model_version,
            "trace_id": trace_id or new_trace_id(),
        }

    def bind(self, **fields: Any) -> "RunLogger":
        """Return a child logger with updated context fields."""
        merged = {**self.context, **fields}
        return RunLogger(**merged)  # type: ignore[arg-type]

    def _log(self, level: int, message: str, **fields: Any) -> None:
        self._logger.log(
            level, message,
            extra={**self.context, "extra_fields": fields},
        )

    def debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._log(logging.ERROR, message, **fields)


__all__ = ["JsonFormatter", "configure_logging", "new_trace_id", "RunLogger"]
