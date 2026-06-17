"""Commentators — turn vision-routine output into commentary events.

A *commentator* is the bridge from the detection world (numbers, boxes) to the
observability world (commentary, attributes, metrics). It is the parallel,
**orthogonal** layer: it *reads* the JSON envelopes produced by
``apps.analytics.routines`` and emits :class:`CommentaryEvent` objects — it
never mutates the detection payload or the routines themselves.

This module ships the deterministic :class:`TemplateCommentator` (no LLM, no API
keys, fully reproducible — ideal for CI and offline runs). The interface is
designed so a future ``AzureVLMCommentator`` (Phase 4) can drop in behind the
same :class:`Commentator` base class and :func:`get_commentator` factory.

Like :mod:`apps.observability.schema`, this module is Django-free and
stdlib-only.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional

from .schema import CommentaryEvent, derive_metrics, new_span_id


# ---------------------------------------------------------------------------
# Context passed into a commentator for one frame/segment
# ---------------------------------------------------------------------------


class CommentaryContext:
    """Carries the correlation context for the events being generated."""

    def __init__(
        self,
        trace_id: str,
        *,
        video_id: Optional[int] = None,
        analysis_id: Optional[int] = None,
        frame_index: Optional[int] = None,
        parent_span_id: Optional[str] = None,
        fps: Optional[float] = None,
        frame_step: Optional[int] = None,
    ) -> None:
        self.trace_id = trace_id
        self.video_id = video_id
        self.analysis_id = analysis_id
        self.frame_index = frame_index
        self.parent_span_id = parent_span_id
        self.fps = fps
        # Number of source frames a sampled frame represents (the sampling
        # stride). Used to bound the temporal segment a commentary covers.
        self.frame_step = frame_step

    def correlation_key(self) -> str:
        """Stable key linking all events about the same video frame.

        This is the join key between an upstream detection, the commentary
        derived from it, and any later agent commentary about the same moment.
        """

        parts = []
        if self.video_id is not None:
            parts.append("video:%s" % self.video_id)
        if self.frame_index is not None:
            parts.append("frame:%s" % self.frame_index)
        return "|".join(parts) or self.trace_id

    def segment_seconds(self) -> Optional[float]:
        """Frame index → segment start in seconds, when fps is known (else ``None``)."""

        if self.frame_index is None or not self.fps:
            return None
        return self.frame_index / float(self.fps)

    def segment_end_seconds(self) -> Optional[float]:
        """End of the temporal segment a sampled frame covers, in seconds.

        A frame sampled at absolute index ``i`` with stride ``frame_step``
        represents ``[i/fps, (i + step)/fps)``. When the stride is unknown the
        segment is treated as a single frame (``step = 1``). Returns ``None``
        when start can't be computed (no fps / no frame index).
        """

        start = self.segment_seconds()
        if start is None:
            return None
        span = self.frame_step if self.frame_step and self.frame_step > 0 else 1
        return (self.frame_index + span) / float(self.fps)


# ---------------------------------------------------------------------------
# Per-routine template functions: (routine_name, result, ctx) -> (text, attrs)
# ---------------------------------------------------------------------------

TemplateFn = Callable[[str, Dict[str, Any], CommentaryContext], "tuple"]

_TEMPLATES: Dict[str, TemplateFn] = {}


def template(routine_name: str) -> Callable[[TemplateFn], TemplateFn]:
    """Register a commentary template for a given routine name."""

    def _wrap(fn: TemplateFn) -> TemplateFn:
        _TEMPLATES[routine_name] = fn
        return fn

    return _wrap


def _labels_phrase(detections: List[Dict[str, Any]]) -> str:
    """Human-readable summary of the distinct labels present, e.g. "2 car, 1 person"."""

    counts: Dict[str, int] = {}
    for d in detections:
        lbl = str(d.get("label", "object"))
        counts[lbl] = counts.get(lbl, 0) + 1
    if not counts:
        return "no objects"
    return ", ".join("%d %s" % (n, lbl) for lbl, n in sorted(counts.items()))


@template("color_detection")
@template("threshold_detection")
def _tmpl_detection(routine: str, result: Dict[str, Any], ctx: CommentaryContext):
    dets = result.get("detections", [])
    where = "frame %s" % ctx.frame_index if ctx.frame_index is not None else "the frame"
    text = "%s detected %s in %s." % (routine, _labels_phrase(dets), where)
    return text, {"labels": sorted({str(d.get("label", "object")) for d in dets})}


@template("zone_counting")
def _tmpl_zone(routine: str, result: Dict[str, Any], ctx: CommentaryContext):
    counts = result.get("summary", {}).get("counts", result.get("counts", {}))
    if counts:
        breakdown = ", ".join("%s=%s" % (z, n) for z, n in sorted(counts.items()))
        text = "Zone occupancy — %s." % breakdown
    else:
        text = "Zone counting produced no occupied zones."
    return text, {"zones": counts}


@template("parking_occupancy")
def _tmpl_parking(routine: str, result: Dict[str, Any], ctx: CommentaryContext):
    spots = result.get("summary", {}).get("spots", result.get("spots", []))
    occupied = sum(1 for s in spots if s.get("occupied"))
    total = len(spots)
    text = "Parking occupancy: %d of %d spots occupied." % (occupied, total)
    return text, {"occupied": occupied, "total_spots": total}


def _tmpl_default(routine: str, result: Dict[str, Any], ctx: CommentaryContext):
    dets = result.get("detections", [])
    text = "Routine '%s' produced %d detection(s)." % (routine, len(dets))
    return text, {}


# ---------------------------------------------------------------------------
# Commentator base + template implementation
# ---------------------------------------------------------------------------


class Commentator(abc.ABC):
    """Strategy that converts a single routine result into commentary events."""

    name = "base"

    @abc.abstractmethod
    def comment_on_result(
        self, routine: str, result: Dict[str, Any], ctx: CommentaryContext
    ) -> List[CommentaryEvent]:
        """Return zero or more :class:`CommentaryEvent` for one routine result."""


class TemplateCommentator(Commentator):
    """Deterministic, rule-based commentary. No model calls, fully reproducible."""

    name = "template"

    def comment_on_result(
        self, routine: str, result: Dict[str, Any], ctx: CommentaryContext
    ) -> List[CommentaryEvent]:
        if "error" in result:
            text = "Routine '%s' failed: %s" % (routine, result["error"])
            attrs: Dict[str, Any] = {"error": result["error"]}
            metrics: Dict[str, float] = {}
        else:
            fn = _TEMPLATES.get(routine, _tmpl_default)
            text, attrs = fn(routine, result, ctx)
            metrics = derive_metrics(result.get("detections", []))
            # Fold any scalar summary numbers into metrics for query-time rollups.
            for k, v in (result.get("summary", {}) or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics.setdefault(k, float(v))

        event = CommentaryEvent(
            commentary=text,
            attributes=attrs,
            metrics=metrics,
            metadata={"routine": routine},
            trace_id=ctx.trace_id,
            span_id=new_span_id(),
            parent_span_id=ctx.parent_span_id,
            correlation_key=ctx.correlation_key(),
            source="routine:%s" % routine,
            video_id=ctx.video_id,
            analysis_id=ctx.analysis_id,
            frame_index=ctx.frame_index,
            segment_start=ctx.segment_seconds(),
            segment_end=ctx.segment_end_seconds(),
        )
        return [event]


_COMMENTATORS: Dict[str, Callable[[], Commentator]] = {
    "template": TemplateCommentator,
}


def get_commentator(name: str = "template") -> Commentator:
    """Return a commentator instance by name (defaults to the template one)."""

    if name == "vlm":
        # Lazy import keeps the LLM dependency out of the core import path and
        # avoids a circular import (vlm imports this module's base class).
        from .vlm import AzureVLMCommentator

        return AzureVLMCommentator()
    try:
        return _COMMENTATORS[name]()
    except KeyError as exc:  # pragma: no cover - trivial
        raise KeyError(
            "Unknown commentator '%s'. Available: %s"
            % (name, sorted(_COMMENTATORS) + ["vlm"])
        ) from exc


# ---------------------------------------------------------------------------
# Bridge: aggregated routine results -> a flat list of commentary events
# ---------------------------------------------------------------------------


def events_from_results(
    results: Dict[str, Any],
    *,
    trace_id: str,
    video_id: Optional[int] = None,
    analysis_id: Optional[int] = None,
    fps: Optional[float] = None,
    frame_step: Optional[int] = None,
    commentator: Optional[Commentator] = None,
) -> List[CommentaryEvent]:
    """Walk the aggregated routine output and produce commentary events.

    ``results`` is the structure returned by
    ``apps.analytics.tasks._run_routines_on_video`` —
    ``{"frame_routines": {name: [{"frame", "result"}]}, "video_routines": {...}}``.

    The function is pure: it reads the detection envelopes and emits events. It
    is the single bridge the Celery task calls, keeping the emission hook to one
    line at the call site.
    """

    commentator = commentator or TemplateCommentator()
    events: List[CommentaryEvent] = []

    # Frame-level routines: one event per (routine, sampled frame).
    for routine, entries in (results.get("frame_routines") or {}).items():
        for entry in entries:
            frame_index = entry.get("frame")
            result = entry.get("result", {})
            ctx = CommentaryContext(
                trace_id,
                video_id=video_id,
                analysis_id=analysis_id,
                frame_index=frame_index,
                fps=fps,
                frame_step=frame_step,
            )
            events.extend(commentator.comment_on_result(routine, result, ctx))

    # Video-level routines: a per-frame event where the routine reports activity,
    # plus a single rollup summary event for the whole pass.
    for routine, vres in (results.get("video_routines") or {}).items():
        per_frame = vres.get("per_frame", []) or []
        total_moving = 0
        for fr in per_frame:
            moving = fr.get("moving_objects", []) or []
            if not moving:
                continue
            total_moving += len(moving)
            frame_index = fr.get("frame")
            ctx = CommentaryContext(
                trace_id, video_id=video_id, analysis_id=analysis_id,
                frame_index=frame_index, fps=fps, frame_step=frame_step,
            )
            events.append(
                CommentaryEvent(
                    commentary="%s observed %d moving object(s) in frame %s."
                    % (routine, len(moving), frame_index),
                    attributes={"moving": len(moving)},
                    metrics={"count": float(len(moving))},
                    metadata={"routine": routine, "level": "video"},
                    trace_id=trace_id,
                    span_id=new_span_id(),
                    correlation_key=ctx.correlation_key(),
                    source="routine:%s" % routine,
                    video_id=video_id,
                    analysis_id=analysis_id,
                    frame_index=frame_index,
                    segment_start=ctx.segment_seconds(),
                    segment_end=ctx.segment_end_seconds(),
                )
            )

        # Rollup event summarising the whole video-level pass.
        events.append(
            CommentaryEvent(
                commentary="%s processed %d frame(s); %d total moving detection(s)."
                % (routine, vres.get("frames_processed", len(per_frame)), total_moving),
                attributes={"frames_processed": vres.get("frames_processed", len(per_frame))},
                metrics={
                    "frames_processed": float(vres.get("frames_processed", len(per_frame))),
                    "total_moving": float(total_moving),
                },
                metadata={"routine": routine, "level": "video", "rollup": True},
                trace_id=trace_id,
                span_id=new_span_id(),
                correlation_key="video:%s|routine:%s" % (video_id, routine),
                source="routine:%s" % routine,
                video_id=video_id,
                analysis_id=analysis_id,
            )
        )

    return events
