"""Semantic aggregation agent (Phase 4).

The :class:`SemanticAggregatorAgent` consumes *low-level* commentary events
(per-frame / per-routine) and emits a *higher-level* semantic event — a scene or
trace summary — correlated back to the same ``trace_id`` and the children's span
ids. This realises the framework's agentic layer: lower-level observations roll
up into higher-level narrative, with the correlation graph preserved.

The roll-up digest is pure (and unit-tested); the narrative text comes from an
injectable :class:`~apps.observability.llm.LLMClient` with a deterministic
fallback, so the agent runs offline and never fails an emission.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .llm import LLMClient, get_llm_client
from .schema import CommentaryEvent, new_span_id, new_trace_id

DEFAULT_SYSTEM_PROMPT = (
    "You are a senior drone-video analyst. Given a digest of per-frame "
    "observations, write a short, high-level summary (2-3 sentences) of what "
    "happened across the segment. Be factual and concise."
)


def digest_events(events: List[CommentaryEvent]) -> Dict[str, Any]:
    """Summarise a list of low-level events into a compact, pure digest."""

    sources: Dict[str, int] = {}
    frames = set()
    metric_totals: Dict[str, float] = {}
    for e in events:
        sources[e.source] = sources.get(e.source, 0) + 1
        if e.frame_index is not None:
            frames.add(e.frame_index)
        for k, v in (e.metrics or {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            metric_totals[k] = metric_totals.get(k, 0.0) + float(v)
    return {
        "event_count": len(events),
        "sources": sources,
        "frames_covered": len(frames),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "metric_totals": metric_totals,
    }


def _digest_text(digest: Dict[str, Any]) -> str:
    """Render the digest as a compact prompt/fallback string (deterministic)."""

    src = ", ".join("%s=%d" % (s, n) for s, n in sorted(digest["sources"].items()))
    metrics = ", ".join("%s=%g" % (k, v) for k, v in sorted(digest["metric_totals"].items()))
    return (
        "Events: %d across %d frame(s) [%s..%s]. Sources: %s. Totals: %s."
        % (
            digest["event_count"], digest["frames_covered"],
            digest["frame_min"], digest["frame_max"], src or "none", metrics or "none",
        )
    )


class SemanticAggregatorAgent:
    """Rolls low-level commentary events up into a higher-level semantic event."""

    name = "semantic"

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.system_prompt = system_prompt

    def summarize(
        self,
        events: Iterable[CommentaryEvent],
        *,
        trace_id: Optional[str] = None,
        video_id: Optional[int] = None,
        analysis_id: Optional[int] = None,
        scope: str = "video",
    ) -> List[CommentaryEvent]:
        """Produce a single higher-level semantic event from ``events``.

        The new event shares the underlying ``trace_id`` and records the child
        span ids it was derived from, keeping the correlation graph intact.
        """

        events = list(events)
        if not events:
            return []

        trace_id = trace_id or events[0].trace_id or new_trace_id()
        digest = digest_events(events)
        digest_text = _digest_text(digest)

        try:
            text = self.llm.complete(digest_text, system=self.system_prompt)
        except Exception:  # noqa: BLE001 - never fail the roll-up
            text = "Scene summary — %s" % digest_text

        correlation_key = (
            "video:%s|scope:%s" % (video_id, scope)
            if video_id is not None
            else "trace:%s|scope:%s" % (trace_id, scope)
        )
        event = CommentaryEvent(
            commentary=text,
            attributes={
                "scope": scope,
                "sources": digest["sources"],
                "derived_from": digest["event_count"],
                # Bounded list so a huge trace doesn't bloat one row.
                "derived_from_spans": [e.span_id for e in events][:64],
                "generated_by": self.llm.name,
            },
            metrics={
                "events_summarized": float(digest["event_count"]),
                "frames_covered": float(digest["frames_covered"]),
                **{("total_%s" % k): v for k, v in digest["metric_totals"].items()},
            },
            metadata={"agent": "semantic", "level": "scene", "rollup": True},
            trace_id=trace_id,
            span_id=new_span_id(),
            correlation_key=correlation_key,
            source="agent:semantic",
            video_id=video_id,
            analysis_id=analysis_id,
        )
        return [event]
