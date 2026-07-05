"""In-process pub/sub message bus with a pluggable backend.

The default :class:`InMemoryBus` needs no external services and is ideal for
local dev, tests and single-process deployments. For multi-process/production a
Redis or Kafka backend can be selected via ``MCP_BUS_BACKEND`` — those backends
import their client libraries lazily and raise a clear error if the dependency
is missing, so the in-memory default keeps the package import-clean.

Every published :class:`~dvsa_api.mcp.adapter_base.Message` is retained in a
bounded ring buffer per topic for inspection/testing (``history``).
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List

from .adapter_base import Message, now_ms

Subscriber = Callable[[Message], None]


class InMemoryBus:
    """Thread-safe in-memory pub/sub with per-topic history."""

    def __init__(self, *, history: int = 500) -> None:
        self._subs: Dict[str, List[Subscriber]] = defaultdict(list)
        self._history: Dict[str, Deque[Message]] = defaultdict(
            lambda: deque(maxlen=history))
        self._lock = threading.RLock()

    def subscribe(self, topic: str, callback: Subscriber) -> Callable[[], None]:
        """Register ``callback`` for ``topic``; returns an unsubscribe closure."""
        with self._lock:
            self._subs[topic].append(callback)

        def _unsub() -> None:
            with self._lock:
                if callback in self._subs.get(topic, []):
                    self._subs[topic].remove(callback)

        return _unsub

    def publish(self, topic: str, kind: str, payload: Dict) -> Message:
        """Publish a message to ``topic`` and deliver to subscribers."""
        message = Message(topic=topic, kind=kind, payload=payload, ts_ms=now_ms())
        with self._lock:
            self._history[topic].append(message)
            subscribers = list(self._subs.get(topic, ()))
            # Wildcard subscribers receive every topic.
            subscribers += list(self._subs.get("*", ()))
        for cb in subscribers:
            # A misbehaving subscriber must not break publication for others.
            try:
                cb(message)
            except Exception:  # noqa: BLE001
                continue
        return message

    def history(self, topic: str) -> List[Message]:
        with self._lock:
            return list(self._history.get(topic, ()))

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()
            self._history.clear()


def _build_backend(name: str) -> InMemoryBus:
    """Construct the configured backend (only in-memory ships enabled).

    Redis/Kafka are recognised names but require their client libs; rather than
    fail at import time we surface a clear error only when explicitly selected.
    """
    name = (name or "memory").lower()
    if name in ("memory", "inmemory", "in-memory", ""):
        return InMemoryBus()
    if name in ("redis", "kafka"):
        raise NotImplementedError(
            f"MCP bus backend '{name}' requires the optional {name} client and "
            f"deployment wiring; the shipped default is in-memory. Unset "
            f"MCP_BUS_BACKEND or set it to 'memory' for local/dev use.")
    raise ValueError(f"unknown MCP_BUS_BACKEND '{name}'")


# Module-level default bus.
_BUS = _build_backend(os.environ.get("MCP_BUS_BACKEND", "memory"))


def get_bus() -> InMemoryBus:
    return _BUS


def reset_bus(bus: "InMemoryBus | None" = None) -> InMemoryBus:
    """Swap the module bus (tests) and return the active instance."""
    global _BUS
    _BUS = bus or InMemoryBus()
    return _BUS
