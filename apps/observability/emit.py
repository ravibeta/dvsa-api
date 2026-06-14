"""High-level emission helpers — the Django-aware glue.

These functions tie the pure core (commentator + sink + schema) to Django
settings. They are the *only* place the rest of the app needs to call:

* :func:`emit_analysis_commentary` — the single hook the Celery task invokes
  after a vision run (Phase 1).
* :func:`ingest_event` — persists a custom event posted to the API (Phase 2).

Both honour ``COMMENTARY_ENABLED`` / ``COMMENTARY_SINK`` / ``COMMENTARY_COMMENTATOR``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from django.conf import settings

from .commentator import events_from_results, get_commentator
from .schema import CommentaryEvent, make_event, new_trace_id
from .sinks import DjangoModelSink, get_sink

logger = logging.getLogger("apps.observability")


def emit_analysis_commentary(
    results: Dict[str, Any],
    *,
    video_id: Optional[int],
    analysis_id: Optional[int],
    fps: Optional[float] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate and persist commentary for one analysis run.

    Reads the aggregated routine ``results`` (the structure produced by
    ``apps.analytics.tasks._run_routines_on_video``), converts them to wide
    commentary events with the configured commentator, and emits them through
    the configured sink. Returns a small summary for logging.

    This is intentionally side-effect-only and defensive at its single call
    site: commentary must never break the vision pipeline (orthogonality).
    """

    trace_id = trace_id or new_trace_id()
    commentator = get_commentator(getattr(settings, "COMMENTARY_COMMENTATOR", "template"))
    events = events_from_results(
        results,
        trace_id=trace_id,
        video_id=video_id,
        analysis_id=analysis_id,
        fps=fps,
        commentator=commentator,
    )

    sink = get_sink()
    emitted = sink.emit_many(events)
    sink.flush()
    return {"trace_id": trace_id, "emitted": emitted}


def ingest_event(validated: Dict[str, Any]) -> CommentaryEvent:
    """Persist a single custom commentary event and return the built event.

    ``validated`` is the output of ``CommentaryEventIngestSerializer``. Missing
    identity/timestamp fields are filled by :func:`make_event`. Custom events are
    always written to the DB sink so they are durable regardless of the
    ``COMMENTARY_SINK`` setting used for routine commentary.
    """

    event = make_event(**{k: v for k, v in validated.items() if v is not None})
    sink = DjangoModelSink()
    sink.emit(event)
    sink.flush()
    return event


def _record_to_event(rec) -> CommentaryEvent:
    """Rehydrate a stored ``CommentaryEventRecord`` into a ``CommentaryEvent``."""

    return CommentaryEvent(
        commentary=rec.commentary,
        attributes=rec.attributes or {},
        metrics=rec.metrics or {},
        metadata=rec.metadata or {},
        event_id=rec.event_id,
        trace_id=rec.trace_id,
        span_id=rec.span_id,
        parent_span_id=rec.parent_span_id,
        correlation_key=rec.correlation_key,
        timestamp=rec.timestamp.isoformat() if rec.timestamp else None,
        source=rec.source,
        video_id=rec.video_id,
        analysis_id=rec.analysis_id,
        frame_index=rec.frame_index,
        segment_start=rec.segment_start,
        segment_end=rec.segment_end,
        schema_version=rec.schema_version,
    )


def run_semantic_agent(
    *,
    video_id: Optional[int] = None,
    trace_id: Optional[str] = None,
    analysis_id: Optional[int] = None,
    scope: str = "video",
    limit: int = 5000,
) -> Dict[str, Any]:
    """Run the semantic aggregation agent over stored low-level events.

    Selects events by ``video_id`` or ``trace_id`` (its own ``agent:semantic``
    output is excluded so summaries don't recursively summarise themselves),
    produces a higher-level event, persists it, and returns a summary. The new
    event shares the source events' ``trace_id`` and links their span ids.
    """

    from .agents import SemanticAggregatorAgent
    from .llm import get_llm_client
    from .models import CommentaryEventRecord

    qs = CommentaryEventRecord.objects.exclude(source="agent:semantic")
    if video_id is not None:
        qs = qs.filter(video_id=video_id)
    if trace_id:
        qs = qs.filter(trace_id=trace_id)
    if analysis_id is not None:
        qs = qs.filter(analysis_id=analysis_id)

    records = list(qs.order_by("timestamp")[:limit])
    if not records:
        return {"summarized": 0, "events": []}

    source_events = [_record_to_event(r) for r in records]
    agent = SemanticAggregatorAgent(get_llm_client())
    new_events = agent.summarize(
        source_events,
        trace_id=trace_id,
        video_id=video_id,
        analysis_id=analysis_id,
        scope=scope,
    )

    sink = DjangoModelSink()
    sink.emit_many(new_events)
    sink.flush()
    return {
        "summarized": len(source_events),
        "events": [e.to_dict() for e in new_events],
    }
