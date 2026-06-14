"""Commentary event schema — the wide, first-class telemetry record.

This module is the heart of the commentary-driven observability layer described
in *"From Video to Commentary"*. It is deliberately **Django-free and
stdlib-only** so it can be imported, unit-tested and reused anywhere (Celery
workers, agents, external scripts) exactly like the routines layer in
``apps.analytics.routines``.

A :class:`CommentaryEvent` is a *single wide row* per frame or temporal segment
that carries, side by side:

* ``commentary``  — natural-language text describing what happened,
* ``attributes``  — open-ended semantic attributes (labels, zones, verdicts),
* ``metrics``     — derived numeric measurements (counts, areas, scores),
* ``metadata``    — contextual metadata (drone pose, gps, model versions),

together with the correlation triad (``trace_id`` / ``span_id`` /
``correlation_key``) that links it to the upstream detection run, sibling
commentary and downstream analytics.

Design principles honoured here
-------------------------------
* **Commentary as first-class telemetry** — text + attributes + metrics +
  metadata live in one event, not bolted onto the detection payload.
* **Query-time aggregation** — events are stored raw and wide; nothing is
  pre-aggregated at write time.
* **Extensibility** — ``attributes`` / ``metrics`` / ``metadata`` are free-form
  JSON maps, so users/agents inject arbitrary fields without a schema change.
* **OTel compatibility** — :meth:`CommentaryEvent.to_otel_log_record` projects
  the wide event onto the OpenTelemetry log-record shape (Body/Attributes/
  TraceId/SpanId/Timestamp), the seam Phase 3 will export over OTLP.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Identity / correlation helpers (OpenTelemetry-compatible widths)
# ---------------------------------------------------------------------------
#
# OTel trace ids are 16 bytes (32 hex chars) and span ids are 8 bytes
# (16 hex chars). We mirror those widths so the same ids can later be emitted
# verbatim through an OTLP exporter without remapping.


def new_trace_id() -> str:
    """Return a fresh 32-hex-char trace id (one per analysis run)."""

    return uuid.uuid4().hex


def new_span_id() -> str:
    """Return a fresh 16-hex-char span id (one per routine-on-frame / emit)."""

    return uuid.uuid4().hex[:16]


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# The wide event
# ---------------------------------------------------------------------------


@dataclass
class CommentaryEvent:
    """A single wide commentary/observability event.

    Every field except the four payload maps is a scalar so the event flattens
    cleanly into a columnar store; ``attributes`` / ``metrics`` / ``metadata``
    absorb everything open-ended, which is what keeps the schema stable while
    remaining infinitely extensible.
    """

    # --- the commentary payload ------------------------------------------- #
    commentary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- identity & correlation ------------------------------------------- #
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str = field(default_factory=new_trace_id)
    span_id: str = field(default_factory=new_span_id)
    parent_span_id: Optional[str] = None
    correlation_key: str = ""

    # --- temporal / source context ---------------------------------------- #
    timestamp: str = field(default_factory=utc_now_iso)
    source: str = "external"  # e.g. "routine:color_detection", "agent:vlm"
    video_id: Optional[int] = None
    analysis_id: Optional[int] = None
    frame_index: Optional[int] = None
    segment_start: Optional[float] = None  # seconds into the video
    segment_end: Optional[float] = None

    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation (stable field order)."""

        return {
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "correlation_key": self.correlation_key,
            "timestamp": self.timestamp,
            "source": self.source,
            "video_id": self.video_id,
            "analysis_id": self.analysis_id,
            "frame_index": self.frame_index,
            "segment_start": self.segment_start,
            "segment_end": self.segment_end,
            "commentary": self.commentary,
            "attributes": self.attributes,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommentaryEvent":
        """Build an event from a (possibly partial) dict.

        Unknown keys are ignored and missing keys fall back to defaults, which
        is what makes **custom event injection** safe: an external caller need
        only supply ``commentary`` (and whatever attributes it cares about) and
        the server fills in identity/timestamp.
        """

        known = {
            "commentary", "attributes", "metrics", "metadata",
            "event_id", "trace_id", "span_id", "parent_span_id",
            "correlation_key", "timestamp", "source", "video_id",
            "analysis_id", "frame_index", "segment_start", "segment_end",
            "schema_version",
        }
        kwargs = {k: v for k, v in (data or {}).items() if k in known and v is not None}
        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    # MELT / OpenTelemetry projection (Phase 3 export seam)
    # ------------------------------------------------------------------ #
    def to_otel_log_record(self) -> Dict[str, Any]:
        """Project the wide event onto the OpenTelemetry log-record shape.

        Metrics and metadata are flattened into the OTel ``Attributes`` map
        under ``metric.*`` / ``meta.*`` prefixes so downstream MELT backends can
        index them; the derived metrics remain individually queryable as OTel
        metrics in Phase 3 via the same prefixes.
        """

        attributes: Dict[str, Any] = dict(self.attributes)
        attributes["source"] = self.source
        if self.correlation_key:
            attributes["correlation_key"] = self.correlation_key
        if self.frame_index is not None:
            attributes["frame_index"] = self.frame_index
        if self.video_id is not None:
            attributes["video_id"] = self.video_id
        if self.analysis_id is not None:
            attributes["analysis_id"] = self.analysis_id
        for k, v in self.metrics.items():
            attributes["metric.%s" % k] = v
        for k, v in self.metadata.items():
            attributes["meta.%s" % k] = v

        return {
            "Timestamp": self.timestamp,
            "TraceId": self.trace_id,
            "SpanId": self.span_id,
            "SeverityText": "INFO",
            "Body": self.commentary,
            "Attributes": attributes,
        }


def make_event(**overrides: Any) -> CommentaryEvent:
    """Convenience constructor that applies ``overrides`` on top of defaults."""

    return CommentaryEvent.from_dict(overrides)


def derive_metrics(detections: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute generic derived metrics from a list of detection dicts.

    Works on the JSON detection envelope produced by
    ``apps.analytics.routines.base.Detection.to_dict`` (keys ``area`` /
    ``score`` / ``label``). Returns counts, total/mean area and mean/max score —
    the numeric backbone every routine's commentary shares.
    """

    count = len(detections)
    if count == 0:
        return {"count": 0.0, "total_area": 0.0, "mean_area": 0.0,
                "mean_score": 0.0, "max_score": 0.0}
    areas = [float(d.get("area", 0.0) or 0.0) for d in detections]
    scores = [float(d.get("score", 0.0) or 0.0) for d in detections]
    return {
        "count": float(count),
        "total_area": float(sum(areas)),
        "mean_area": float(sum(areas) / count),
        "mean_score": float(sum(scores) / count),
        "max_score": float(max(scores)),
    }
