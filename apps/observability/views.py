"""API for the commentary observability layer.

Three endpoints, mirroring the framework principles:

* ``POST  /observability/events/``          — inject a custom commentary event
  (extensibility).
* ``GET   /observability/events/``          — list/filter raw wide events.
* ``GET   /observability/events/aggregate/`` — query-time roll-ups over raw events.
"""

from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .aggregation import aggregate_events
from .emit import ingest_event, run_semantic_agent
from .models import CommentaryEventRecord
from .serializers import (
    AgentSummarizeSerializer,
    CommentaryEventIngestSerializer,
    CommentaryEventRecordSerializer,
)

# Fields a caller may filter the raw event list / aggregation on.
_FILTERABLE = {
    "trace_id", "correlation_key", "source", "video_id", "analysis_id",
    "frame_index", "schema_version",
}


def _query_filters(params) -> dict:
    """Extract supported exact-match filters from query params."""

    out = {}
    for key in _FILTERABLE:
        if key in params:
            out[key] = params.get(key)
    return out


class EventIngestListView(generics.ListAPIView):
    """List raw commentary events (GET) and inject custom ones (POST)."""

    serializer_class = CommentaryEventRecordSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = CommentaryEventRecord.objects.all()
        filters = _query_filters(self.request.query_params)
        if filters:
            qs = qs.filter(**filters)
        return qs

    def post(self, request, *args, **kwargs):
        serializer = CommentaryEventIngestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        event = ingest_event(serializer.validated_data)
        record = CommentaryEventRecord.objects.get(event_id=event.event_id)
        return Response(
            CommentaryEventRecordSerializer(record).data,
            status=status.HTTP_201_CREATED,
        )


class EventAggregateView(APIView):
    """Query-time aggregation over raw events.

    Query params:
        ``group_by``  — field path to group on (e.g. ``source``,
                        ``metadata.routine``). Optional.
        ``metrics``   — comma-separated metric specs (``count``, ``sum:count``,
                        ``avg:mean_score``). Defaults to ``count``.
        plus any of the filterable fields for exact-match filtering.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        params = request.query_params
        group_by = params.get("group_by") or None
        metrics = [m for m in (params.get("metrics", "count").split(",")) if m]
        filters = _query_filters(params)

        # Pull only the columns aggregation needs; aggregate in Python so the
        # same pure engine serves DB, in-memory and (later) OTLP sources.
        rows = CommentaryEventRecord.objects.filter(**filters).values(
            "source", "frame_index", "video_id", "analysis_id",
            "correlation_key", "metrics", "metadata", "attributes",
        )
        try:
            result = aggregate_events(rows, group_by=group_by, metrics=metrics)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(result)


class AgentSummarizeView(APIView):
    """Roll low-level commentary events up into a higher-level semantic event.

    ``POST`` body: ``{"video_id": <int>}`` and/or ``{"trace_id": <hex>}`` plus an
    optional ``scope``. Runs the semantic aggregation agent over the selected
    stored events, persists the summary, and returns it.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = AgentSummarizeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        result = run_semantic_agent(
            video_id=data.get("video_id"),
            trace_id=data.get("trace_id") or None,
            analysis_id=data.get("analysis_id"),
            scope=data.get("scope", "video"),
        )
        code = status.HTTP_201_CREATED if result["events"] else status.HTTP_200_OK
        return Response(result, status=code)
