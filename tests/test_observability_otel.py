"""Unit tests for the OTel/MELT export layer (Phase 3).

These cover the pure OTLP/JSON payload builders and the fan-out/OTel sinks with
an injected fake exporter — no network or OTel SDK required, mirroring the
pure-Python style of ``tests/test_routines.py`` and ``tests/test_observability.py``.
"""

from apps.observability.schema import CommentaryEvent
from apps.observability.otel import (
    OTLPHttpExporter,
    iso_to_unix_nano,
    to_otlp_logs_payload,
    to_otlp_metrics_payload,
    to_otlp_traces_payload,
    _attr_value,
)
from apps.observability.sinks import InMemorySink, OTelSink, TeeSink, get_sink


def _event(**kw):
    base = dict(
        commentary="2 car detected in frame 30",
        attributes={"labels": ["car"]},
        metrics={"count": 2.0, "total_area": 300.0},
        metadata={"routine": "color_detection"},
        trace_id="a" * 32,
        span_id="b" * 16,
        parent_span_id=None,
        correlation_key="video:7|frame:30",
        timestamp="2026-06-13T00:00:00+00:00",
        source="routine:color_detection",
        video_id=7,
        analysis_id=3,
        frame_index=30,
    )
    base.update(kw)
    return CommentaryEvent(**base)


# --------------------------------------------------------------------------- #
# Primitive encoding
# --------------------------------------------------------------------------- #

def test_attr_value_type_encoding():
    assert _attr_value("x") == {"stringValue": "x"}
    assert _attr_value(True) == {"boolValue": True}
    assert _attr_value(5) == {"intValue": "5"}          # int64 as string in OTLP/JSON
    assert _attr_value(1.5) == {"doubleValue": 1.5}
    assert _attr_value(["a", "b"]) == {
        "arrayValue": {"values": [{"stringValue": "a"}, {"stringValue": "b"}]}
    }


def test_iso_to_unix_nano():
    # 2026-06-13T00:00:00Z is a fixed epoch second; nanos = seconds * 1e9.
    nano = iso_to_unix_nano("2026-06-13T00:00:00+00:00")
    assert nano % 1_000_000_000 == 0
    assert iso_to_unix_nano("2026-06-13T00:00:00Z") == nano


# --------------------------------------------------------------------------- #
# Logs payload
# --------------------------------------------------------------------------- #

def test_logs_payload_shape():
    payload = to_otlp_logs_payload([_event()], service_name="dvsa-api")
    rl = payload["resourceLogs"][0]
    assert rl["resource"]["attributes"][0]["key"] == "service.name"
    rec = rl["scopeLogs"][0]["logRecords"][0]
    assert rec["body"]["stringValue"].startswith("2 car")
    assert rec["traceId"] == "a" * 32
    assert rec["spanId"] == "b" * 16
    assert rec["severityText"] == "INFO"
    keys = {a["key"] for a in rec["attributes"]}
    assert "source" in keys and "meta.routine" in keys and "correlation_key" in keys


# --------------------------------------------------------------------------- #
# Metrics payload
# --------------------------------------------------------------------------- #

def test_metrics_payload_gauges():
    payload = to_otlp_metrics_payload([_event()], service_name="dvsa-api")
    metrics = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    names = {m["name"] for m in metrics}
    assert "dvsa.commentary.count" in names
    assert "dvsa.commentary.total_area" in names
    count_metric = next(m for m in metrics if m["name"] == "dvsa.commentary.count")
    dp = count_metric["gauge"]["dataPoints"][0]
    assert dp["asDouble"] == 2.0
    assert "timeUnixNano" in dp


def test_metrics_payload_merges_points_across_events():
    events = [_event(frame_index=0), _event(frame_index=30)]
    payload = to_otlp_metrics_payload(events)
    count_metric = next(
        m for m in payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        if m["name"] == "dvsa.commentary.count"
    )
    assert len(count_metric["gauge"]["dataPoints"]) == 2  # one per event, raw


# --------------------------------------------------------------------------- #
# Traces payload
# --------------------------------------------------------------------------- #

def test_traces_payload_spans():
    payload = to_otlp_traces_payload([_event(parent_span_id="c" * 16)])
    span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == "a" * 32
    assert span["spanId"] == "b" * 16
    assert span["parentSpanId"] == "c" * 16
    assert span["name"] == "routine:color_detection"
    assert span["startTimeUnixNano"] == span["endTimeUnixNano"]  # instantaneous


def test_traces_span_duration_from_segment():
    e = _event(segment_start=1.0, segment_end=2.0)
    span = to_otlp_traces_payload([e])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    dur = int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])
    assert dur == 1_000_000_000  # 1 second


# --------------------------------------------------------------------------- #
# Exporter (no network): build_payloads + signal selection
# --------------------------------------------------------------------------- #

def test_exporter_build_payloads_respects_signal_selection():
    exp = OTLPHttpExporter("http://collector:4318", signals=["logs", "metrics"])
    payloads = exp.build_payloads([_event()])
    assert set(payloads) == {"logs", "metrics"}
    assert "resourceLogs" in payloads["logs"]
    assert "resourceMetrics" in payloads["metrics"]


# --------------------------------------------------------------------------- #
# Sinks: OTelSink with a fake exporter + TeeSink fan-out
# --------------------------------------------------------------------------- #

class _FakeExporter:
    def __init__(self):
        self.exported = []

    def export(self, events):
        self.exported.append(list(events))
        return {"logs": 200}


def test_otel_sink_flushes_buffer_to_exporter():
    fake = _FakeExporter()
    sink = OTelSink(exporter=fake)
    sink.emit_many([_event(), _event()])
    assert fake.exported == []      # nothing sent until flush
    sink.flush()
    assert len(fake.exported) == 1 and len(fake.exported[0]) == 2
    assert sink.last_result == {"logs": 200}
    sink.flush()                    # empty buffer -> no extra export
    assert len(fake.exported) == 1


def test_tee_sink_fans_out():
    a, b = InMemorySink(), InMemorySink()
    tee = TeeSink([a, b])
    tee.emit_many([_event(), _event()])
    tee.flush()
    assert len(a.events) == 2 and len(b.events) == 2


def test_get_sink_comma_list_builds_tee():
    sink = get_sink("memory,null")
    assert isinstance(sink, TeeSink)
    assert len(sink.sinks) == 2


def test_otel_sink_requires_endpoint_without_exporter():
    try:
        OTelSink()  # no exporter, no endpoint
        assert False, "expected ValueError"
    except ValueError:
        pass
