"""Commentary sinks — pluggable destinations for commentary events.

A *sink* is where emitted :class:`~apps.observability.schema.CommentaryEvent`
objects go. The abstraction keeps the emission call site (one line in the
Celery task) decoupled from storage, so swapping DB ⇄ OTLP ⇄ no-op is a config
change, not a code change.

Implementations
---------------
* :class:`NullSink`      — drops everything (default when commentary disabled).
* :class:`InMemorySink`  — keeps events in a list (tests, ad-hoc aggregation).
* :class:`DjangoModelSink` — persists rows to ``CommentaryEventRecord`` (Phase 1
  storage). Django is imported lazily so the rest of this module stays
  importable without a Django context.

``OTelSink`` (Phase 3) will slot in here behind the same interface.
"""

from __future__ import annotations

import abc
from typing import Iterable, List

from .schema import CommentaryEvent


class CommentarySink(abc.ABC):
    """Destination for commentary events."""

    @abc.abstractmethod
    def emit(self, event: CommentaryEvent) -> None:
        """Record a single event."""

    def emit_many(self, events: Iterable[CommentaryEvent]) -> int:
        """Record several events; returns the count emitted."""

        n = 0
        for event in events:
            self.emit(event)
            n += 1
        return n

    def flush(self) -> None:  # pragma: no cover - default no-op
        """Flush any buffered events (no-op unless overridden)."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any resources held by the sink."""


class NullSink(CommentarySink):
    """Discards every event. Used when commentary is disabled."""

    def emit(self, event: CommentaryEvent) -> None:
        return None


class InMemorySink(CommentarySink):
    """Buffers events in memory — handy for tests and query-time aggregation."""

    def __init__(self) -> None:
        self.events: List[CommentaryEvent] = []

    def emit(self, event: CommentaryEvent) -> None:
        self.events.append(event)

    def as_dicts(self) -> List[dict]:
        return [e.to_dict() for e in self.events]


class DjangoModelSink(CommentarySink):
    """Persists events as ``CommentaryEventRecord`` rows.

    Buffers and writes with ``bulk_create`` on :meth:`flush` (and opportunistically
    when the buffer is large) to keep per-event overhead low during a run.
    """

    def __init__(self, batch_size: int = 200) -> None:
        self.batch_size = batch_size
        self._buffer: List[CommentaryEvent] = []

    def emit(self, event: CommentaryEvent) -> None:
        self._buffer.append(event)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        # Lazy imports: keep this module usable without a Django context.
        from django.utils.dateparse import parse_datetime

        from .models import CommentaryEventRecord

        rows = []
        for e in self._buffer:
            rows.append(
                CommentaryEventRecord(
                    event_id=e.event_id,
                    trace_id=e.trace_id,
                    span_id=e.span_id,
                    parent_span_id=e.parent_span_id,
                    correlation_key=e.correlation_key,
                    timestamp=parse_datetime(e.timestamp),
                    source=e.source,
                    video_id=e.video_id,
                    analysis_id=e.analysis_id,
                    frame_index=e.frame_index,
                    segment_start=e.segment_start,
                    segment_end=e.segment_end,
                    commentary=e.commentary,
                    attributes=e.attributes,
                    metrics=e.metrics,
                    metadata=e.metadata,
                    schema_version=e.schema_version,
                )
            )
        CommentaryEventRecord.objects.bulk_create(rows, ignore_conflicts=True)
        self._buffer = []


def get_sink(name: str = None) -> CommentarySink:
    """Return a sink instance by name, reading Django settings when available.

    Resolution order: explicit ``name`` arg → ``COMMENTARY_SINK`` setting →
    ``"null"``. Valid names: ``"null"``, ``"memory"``, ``"db"``.
    """

    if name is None:
        try:
            from django.conf import settings

            name = getattr(settings, "COMMENTARY_SINK", "null")
        except Exception:  # noqa: BLE001 - no Django context -> default
            name = "null"

    name = (name or "null").lower()
    if name in ("null", "none", "off"):
        return NullSink()
    if name in ("memory", "mem"):
        return InMemorySink()
    if name in ("db", "django", "model"):
        return DjangoModelSink()
    raise ValueError("Unknown commentary sink '%s'" % name)
