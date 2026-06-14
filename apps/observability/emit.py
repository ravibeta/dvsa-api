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
