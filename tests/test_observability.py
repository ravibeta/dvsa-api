"""Unit tests for the commentary-driven observability core.

These exercise the pure, Django-free layer (schema, commentator, sinks,
aggregation) directly — no database or Celery required, mirroring the approach
in ``tests/test_routines.py``. The Django model/API shell is a thin persistence
wrapper over the same schema and is covered separately when the full stack runs.
"""

from apps.observability.schema import (
    CommentaryEvent,
    derive_metrics,
    make_event,
    new_span_id,
    new_trace_id,
)
from apps.observability.commentator import (
    CommentaryContext,
    TemplateCommentator,
    events_from_results,
    get_commentator,
)
from apps.observability.sinks import InMemorySink, NullSink, get_sink
from apps.observability.aggregation import aggregate_events


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _detection(label="car", area=100.0, score=0.9):
    return {
        "bbox": [0, 0, 10, 10], "centroid": [5, 5],
        "area": area, "label": label, "score": score, "track_id": None,
    }


def _color_result(n=2):
    dets = [_detection(area=100.0 * (i + 1), score=0.9 - 0.1 * i) for i in range(n)]
    return {"routine": "color_detection", "summary": {"count": n}, "detections": dets}


# --------------------------------------------------------------------------- #
# Schema — identity, widths, round-trip, custom injection
# --------------------------------------------------------------------------- #

def test_trace_and_span_id_widths_are_otel_compatible():
    assert len(new_trace_id()) == 32
    assert len(new_span_id()) == 16


def test_event_round_trips_through_dict():
    e = CommentaryEvent(commentary="hello", attributes={"a": 1}, metrics={"count": 2.0})
    assert CommentaryEvent.from_dict(e.to_dict()).to_dict() == e.to_dict()


def test_make_event_fills_defaults_for_partial_custom_event():
    # Extensibility: a caller supplies only commentary + arbitrary attributes.
    e = make_event(commentary="clear to land", attributes={"verdict": "safe"})
    assert e.source == "external"
    assert e.attributes["verdict"] == "safe"
    assert len(e.trace_id) == 32 and len(e.span_id) == 16


def test_from_dict_ignores_unknown_keys():
    e = CommentaryEvent.from_dict({"commentary": "x", "bogus": 123})
    assert e.commentary == "x"


def test_derive_metrics_empty_and_nonempty():
    assert derive_metrics([])["count"] == 0.0
    m = derive_metrics([_detection(area=100, score=0.8), _detection(area=300, score=0.4)])
    assert m["count"] == 2.0
    assert m["total_area"] == 400.0
    assert abs(m["mean_score"] - 0.6) < 1e-9
    assert m["max_score"] == 0.8


def test_to_otel_log_record_projection():
    e = CommentaryEvent(
        commentary="2 car detected",
        attributes={"labels": ["car"]},
        metrics={"count": 2.0},
        metadata={"routine": "color_detection"},
        correlation_key="video:7|frame:30",
        frame_index=30,
        source="routine:color_detection",
    )
    rec = e.to_otel_log_record()
    assert rec["Body"] == "2 car detected"
    assert rec["TraceId"] == e.trace_id and rec["SpanId"] == e.span_id
    assert rec["Attributes"]["metric.count"] == 2.0
    assert rec["Attributes"]["meta.routine"] == "color_detection"
    assert rec["Attributes"]["correlation_key"] == "video:7|frame:30"


# --------------------------------------------------------------------------- #
# Commentator — template generation + correlation
# --------------------------------------------------------------------------- #

def test_template_commentator_builds_correlated_event():
    trace = new_trace_id()
    ctx = CommentaryContext(trace, video_id=7, analysis_id=3, frame_index=30, fps=30.0)
    events = TemplateCommentator().comment_on_result("color_detection", _color_result(2), ctx)
    assert len(events) == 1
    ev = events[0]
    assert ev.trace_id == trace
    assert ev.source == "routine:color_detection"
    assert ev.correlation_key == "video:7|frame:30"
    assert ev.metrics["count"] == 2.0
    assert ev.metrics["total_area"] == 300.0
    assert ev.segment_start == 1.0  # frame 30 @ 30fps -> 1.0s
    assert "car" in ev.commentary


def test_template_commentator_handles_routine_error():
    ctx = CommentaryContext(new_trace_id(), frame_index=0)
    events = TemplateCommentator().comment_on_result(
        "color_detection", {"routine": "color_detection", "error": "boom"}, ctx
    )
    assert len(events) == 1
    assert "failed" in events[0].commentary
    assert events[0].attributes["error"] == "boom"


def test_get_commentator_default_and_unknown():
    assert isinstance(get_commentator(), TemplateCommentator)
    try:
        get_commentator("does-not-exist")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_events_from_results_covers_frame_and_video_routines():
    trace = new_trace_id()
    results = {
        "frame_routines": {
            "color_detection": [
                {"frame": 0, "result": _color_result(2)},
                {"frame": 30, "result": _color_result(0)},
            ],
        },
        "video_routines": {
            "background_subtraction": {
                "frames_processed": 2,
                "per_frame": [
                    {"frame": 0, "moving_objects": [{"area": 1}, {"area": 2}]},
                    {"frame": 1, "moving_objects": []},
                ],
            },
        },
    }
    events = events_from_results(results, trace_id=trace, video_id=7, analysis_id=3, fps=30.0)
    # 2 frame events + 1 per-frame video event (only frame 0 has motion) + 1 rollup
    assert len(events) == 4
    assert all(e.trace_id == trace for e in events)
    assert sum(1 for e in events if e.metadata.get("rollup")) == 1


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #

def test_in_memory_sink_collects_events():
    sink = InMemorySink()
    n = sink.emit_many([CommentaryEvent(commentary="a"), CommentaryEvent(commentary="b")])
    assert n == 2
    assert len(sink.events) == 2
    assert len(sink.as_dicts()) == 2


def test_null_sink_is_noop():
    NullSink().emit_many([CommentaryEvent(commentary="x")])  # must not raise


def test_get_sink_resolves_names():
    assert isinstance(get_sink("memory"), InMemorySink)
    assert isinstance(get_sink("null"), NullSink)
    try:
        get_sink("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# Query-time aggregation
# --------------------------------------------------------------------------- #

def _sample_events():
    trace = new_trace_id()
    results = {
        "frame_routines": {
            "color_detection": [
                {"frame": 0, "result": _color_result(2)},
                {"frame": 30, "result": _color_result(1)},
            ],
        },
        "video_routines": {},
    }
    return events_from_results(results, trace_id=trace, video_id=7, analysis_id=3)


def test_aggregate_group_by_source_with_metric_specs():
    events = _sample_events()
    agg = aggregate_events(
        events, group_by="source", metrics=["count", "sum:count", "avg:count"]
    )
    assert agg["total_events"] == 2
    grp = agg["groups"]["routine:color_detection"]
    assert grp["events"] == 2
    assert grp["count"] == 2.0          # two events
    assert grp["sum:count"] == 3.0      # 2 + 1 detections
    assert grp["avg:count"] == 1.5


def test_aggregate_with_filter_and_nested_group():
    events = _sample_events()
    agg = aggregate_events(
        events, group_by="metadata.routine", filters={"frame_index": 0}
    )
    assert agg["total_events"] == 1
    assert agg["groups"]["color_detection"]["events"] == 1


def test_aggregate_rejects_unknown_op():
    try:
        aggregate_events([], metrics=["bogus:count"])
        assert False, "expected ValueError"
    except ValueError:
        pass
