"""OpenTelemetry / MELT export for commentary events (Phase 3).

Projects wide :class:`~apps.observability.schema.CommentaryEvent` objects onto the
three OpenTelemetry signals and ships them over **OTLP/HTTP+JSON** — the
vendor-neutral OTel wire protocol — using only the standard library:

* **Logs**    ← the natural-language ``commentary`` (one OTLP log record/event).
* **Metrics** ← the derived numeric ``metrics`` map (one gauge data point each).
* **Traces**  ← each event becomes a span carrying the trace/span correlation.

Why OTLP/HTTP+JSON over the OTel SDK? It is a first-class, spec-defined transport
that needs no extra packages, keeps the payload builders **pure and testable**,
and is accepted by every major collector (OpenTelemetry Collector, Grafana,
Jaeger, Azure Monitor via the OTLP ingestion path). The builders below produce
the exact request bodies for ``/v1/logs``, ``/v1/metrics`` and ``/v1/traces``.

All of the ``to_otlp_*`` functions are pure; only :class:`OTLPHttpExporter`
performs I/O.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .schema import CommentaryEvent

DEFAULT_SERVICE_NAME = "dvsa-api"
DEFAULT_SCOPE = "apps.observability"
METRIC_PREFIX = "dvsa.commentary."


# ---------------------------------------------------------------------------
# OTLP/JSON primitive encoders
# ---------------------------------------------------------------------------


def _attr_value(value: Any) -> Dict[str, Any]:
    """Encode a Python value as an OTLP ``AnyValue``."""

    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}  # OTLP/JSON encodes int64 as string
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_attr_value(v) for v in value]}}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [{"key": str(k), "value": _attr_value(v)} for k, v in value.items()]
            }
        }
    return {"stringValue": str(value)}


def _attributes(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Encode a flat dict as an OTLP ``KeyValue`` list (skips ``None`` values)."""

    return [
        {"key": str(k), "value": _attr_value(v)}
        for k, v in mapping.items()
        if v is not None
    ]


def _resource(service_name: str) -> Dict[str, Any]:
    return {"attributes": _attributes({"service.name": service_name})}


def iso_to_unix_nano(iso: str) -> int:
    """Convert an ISO-8601 timestamp to integer nanoseconds since the epoch."""

    # ``fromisoformat`` handles the ``+00:00`` offset our events emit.
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1_000_000_000)


def _event_otel_attributes(event: CommentaryEvent) -> Dict[str, Any]:
    """Shared attribute set used across all three signals for one event."""

    attrs: Dict[str, Any] = dict(event.attributes)
    attrs["source"] = event.source
    if event.correlation_key:
        attrs["correlation_key"] = event.correlation_key
    if event.video_id is not None:
        attrs["video_id"] = event.video_id
    if event.analysis_id is not None:
        attrs["analysis_id"] = event.analysis_id
    if event.frame_index is not None:
        attrs["frame_index"] = event.frame_index
    for k, v in event.metadata.items():
        attrs["meta.%s" % k] = v
    return attrs


# ---------------------------------------------------------------------------
# Signal builders (pure)
# ---------------------------------------------------------------------------


def to_otlp_logs_payload(
    events: Iterable[CommentaryEvent], *, service_name: str = DEFAULT_SERVICE_NAME
) -> Dict[str, Any]:
    """Build an OTLP ``ExportLogsServiceRequest`` body (commentary → logs)."""

    records = []
    for e in events:
        nano = iso_to_unix_nano(e.timestamp)
        records.append(
            {
                "timeUnixNano": str(nano),
                "observedTimeUnixNano": str(nano),
                "severityNumber": 9,  # INFO
                "severityText": "INFO",
                "body": {"stringValue": e.commentary},
                "attributes": _attributes(_event_otel_attributes(e)),
                "traceId": e.trace_id,
                "spanId": e.span_id,
            }
        )
    return {
        "resourceLogs": [
            {
                "resource": _resource(service_name),
                "scopeLogs": [{"scope": {"name": DEFAULT_SCOPE}, "logRecords": records}],
            }
        ]
    }


def to_otlp_metrics_payload(
    events: Iterable[CommentaryEvent],
    *,
    service_name: str = DEFAULT_SERVICE_NAME,
    prefix: str = METRIC_PREFIX,
) -> Dict[str, Any]:
    """Build an OTLP ``ExportMetricsServiceRequest`` body (derived metrics → gauges).

    Each numeric entry in an event's ``metrics`` map becomes a gauge data point
    named ``<prefix><metric>``; points for the same metric are merged under one
    metric definition. Raw points are exported — aggregation stays at query time.
    """

    points_by_metric: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        nano = iso_to_unix_nano(e.timestamp)
        attrs = _attributes(_event_otel_attributes(e))
        for name, value in (e.metrics or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            points_by_metric.setdefault(prefix + name, []).append(
                {"asDouble": float(value), "timeUnixNano": str(nano), "attributes": attrs}
            )

    metrics = [
        {"name": name, "unit": "1", "gauge": {"dataPoints": points}}
        for name, points in points_by_metric.items()
    ]
    return {
        "resourceMetrics": [
            {
                "resource": _resource(service_name),
                "scopeMetrics": [{"scope": {"name": DEFAULT_SCOPE}, "metrics": metrics}],
            }
        ]
    }


def to_otlp_traces_payload(
    events: Iterable[CommentaryEvent], *, service_name: str = DEFAULT_SERVICE_NAME
) -> Dict[str, Any]:
    """Build an OTLP ``ExportTraceServiceRequest`` body (events → spans)."""

    spans = []
    for e in events:
        start = iso_to_unix_nano(e.timestamp)
        # Use the temporal segment for span duration when available.
        end = start
        if e.segment_start is not None and e.segment_end is not None:
            end = start + int((e.segment_end - e.segment_start) * 1_000_000_000)
        span = {
            "traceId": e.trace_id,
            "spanId": e.span_id,
            "name": e.source,
            "kind": 1,  # SPAN_KIND_INTERNAL
            "startTimeUnixNano": str(start),
            "endTimeUnixNano": str(end),
            "attributes": _attributes(_event_otel_attributes(e)),
        }
        if e.parent_span_id:
            span["parentSpanId"] = e.parent_span_id
        spans.append(span)
    return {
        "resourceSpans": [
            {
                "resource": _resource(service_name),
                "scopeSpans": [{"scope": {"name": DEFAULT_SCOPE}, "spans": spans}],
            }
        ]
    }


# Signal name -> (builder, OTLP path suffix)
_SIGNALS = {
    "logs": (to_otlp_logs_payload, "/v1/logs"),
    "metrics": (to_otlp_metrics_payload, "/v1/metrics"),
    "traces": (to_otlp_traces_payload, "/v1/traces"),
}


# ---------------------------------------------------------------------------
# HTTP exporter (the only I/O in this module)
# ---------------------------------------------------------------------------


class OTLPHttpExporter:
    """POSTs OTLP/JSON payloads to a collector's ``/v1/{logs,metrics,traces}``.

    Depends only on the standard library. ``export`` is best-effort: it returns a
    per-signal status map and never raises for transport errors, so emission can
    never break the analysis pipeline (orthogonality).
    """

    def __init__(
        self,
        endpoint: str,
        *,
        service_name: str = DEFAULT_SERVICE_NAME,
        signals: Optional[Iterable[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.service_name = service_name
        self.signals = list(signals) if signals is not None else ["logs", "metrics", "traces"]
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    def build_payloads(self, events: List[CommentaryEvent]) -> Dict[str, Dict[str, Any]]:
        """Return ``{signal: otlp_payload}`` for the configured signals (pure)."""

        out = {}
        for signal in self.signals:
            builder, _path = _SIGNALS[signal]
            out[signal] = builder(events, service_name=self.service_name)
        return out

    def export(self, events: List[CommentaryEvent]) -> Dict[str, Any]:
        if not events:
            return {}
        results: Dict[str, Any] = {}
        for signal in self.signals:
            builder, path = _SIGNALS[signal]
            payload = builder(events, service_name=self.service_name)
            results[signal] = self._post(path, payload)
        return results

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint + path, data=data, headers=self.headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return getattr(resp, "status", resp.getcode())
        except Exception as exc:  # noqa: BLE001 - best-effort export
            return "error: %s" % exc
