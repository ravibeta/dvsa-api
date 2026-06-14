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
from typing import Iterable, List, Optional

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


class OTelSink(CommentarySink):
    """Exports events to an OpenTelemetry collector over OTLP/HTTP+JSON.

    Buffers events and ships them as Logs/Metrics/Traces on :meth:`flush` via
    :class:`apps.observability.otel.OTLPHttpExporter`. The exporter is injectable
    so the mapping/fan-out can be unit-tested without a live collector.
    """

    def __init__(self, exporter=None, *, endpoint: Optional[str] = None, **exporter_kwargs) -> None:
        if exporter is None:
            from .otel import OTLPHttpExporter

            if not endpoint:
                raise ValueError("OTelSink requires an OTLP endpoint")
            exporter = OTLPHttpExporter(endpoint, **exporter_kwargs)
        self.exporter = exporter
        self._buffer: List[CommentaryEvent] = []
        self.last_result = None

    def emit(self, event: CommentaryEvent) -> None:
        self._buffer.append(event)

    def flush(self) -> None:
        if not self._buffer:
            return
        self.last_result = self.exporter.export(self._buffer)
        self._buffer = []


class TeeSink(CommentarySink):
    """Fan-out sink: writes every event to several child sinks.

    The common MELT pattern is to persist commentary to the DB *and* export it to
    a collector simultaneously (``COMMENTARY_SINK="db,otel"``).
    """

    def __init__(self, sinks: Iterable[CommentarySink]) -> None:
        self.sinks: List[CommentarySink] = list(sinks)

    def emit(self, event: CommentaryEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)

    def flush(self) -> None:
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


def _build_single_sink(name: str) -> CommentarySink:
    name = (name or "null").strip().lower()
    if name in ("null", "none", "off"):
        return NullSink()
    if name in ("memory", "mem"):
        return InMemorySink()
    if name in ("db", "django", "model"):
        return DjangoModelSink()
    if name in ("otel", "otlp"):
        # Configure from Django settings; raises if no endpoint is set.
        endpoint = None
        service_name = "dvsa-api"
        signals = None
        try:
            from django.conf import settings

            endpoint = getattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", None)
            service_name = getattr(settings, "OTEL_SERVICE_NAME", service_name)
            raw_signals = getattr(settings, "COMMENTARY_OTEL_SIGNALS", None)
            if raw_signals:
                signals = [s.strip() for s in raw_signals.split(",") if s.strip()]
        except Exception:  # noqa: BLE001 - no Django context
            pass
        return OTelSink(endpoint=endpoint, service_name=service_name, signals=signals)
    raise ValueError("Unknown commentary sink '%s'" % name)


def get_sink(name: str = None) -> CommentarySink:
    """Return a sink by name, reading Django settings when available.

    Resolution order: explicit ``name`` arg → ``COMMENTARY_SINK`` setting →
    ``"null"``. A comma-separated value (e.g. ``"db,otel"``) yields a
    :class:`TeeSink` fanning out to each. Valid names: ``"null"``, ``"memory"``,
    ``"db"``, ``"otel"``.
    """

    if name is None:
        try:
            from django.conf import settings

            name = getattr(settings, "COMMENTARY_SINK", "null")
        except Exception:  # noqa: BLE001 - no Django context -> default
            name = "null"

    parts = [p for p in (name or "null").split(",") if p.strip()]
    if len(parts) > 1:
        return TeeSink(_build_single_sink(p) for p in parts)
    return _build_single_sink(parts[0] if parts else "null")
