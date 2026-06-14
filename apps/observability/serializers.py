"""Serializers for commentary-event ingestion and read-back."""

from rest_framework import serializers

from .models import CommentaryEventRecord


class CommentaryEventRecordSerializer(serializers.ModelSerializer):
    """Read serializer for stored commentary events."""

    class Meta:
        model = CommentaryEventRecord
        fields = [
            "id", "event_id", "trace_id", "span_id", "parent_span_id",
            "correlation_key", "timestamp", "source", "video_id", "analysis_id",
            "frame_index", "segment_start", "segment_end", "commentary",
            "attributes", "metrics", "metadata", "schema_version", "created_at",
        ]
        read_only_fields = fields


class CommentaryEventIngestSerializer(serializers.Serializer):
    """Validate a custom commentary event injected by a user/agent/external system.

    Only ``commentary`` is conceptually required; identity and timestamp are
    filled server-side when omitted (see :func:`apps.observability.emit.ingest_event`).
    This is the **extensibility** seam: arbitrary ``attributes`` / ``metrics`` /
    ``metadata`` are accepted without any schema change.
    """

    commentary = serializers.CharField(allow_blank=True, default="")
    attributes = serializers.DictField(required=False, default=dict)
    metrics = serializers.DictField(
        required=False, default=dict, child=serializers.FloatField()
    )
    metadata = serializers.DictField(required=False, default=dict)

    # Optional correlation / context — server fills sensible defaults.
    trace_id = serializers.CharField(required=False, allow_blank=True)
    span_id = serializers.CharField(required=False, allow_blank=True)
    parent_span_id = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    correlation_key = serializers.CharField(required=False, allow_blank=True)
    source = serializers.CharField(required=False, default="external")
    video_id = serializers.IntegerField(required=False, allow_null=True)
    analysis_id = serializers.IntegerField(required=False, allow_null=True)
    frame_index = serializers.IntegerField(required=False, allow_null=True)
    segment_start = serializers.FloatField(required=False, allow_null=True)
    segment_end = serializers.FloatField(required=False, allow_null=True)


class AgentSummarizeSerializer(serializers.Serializer):
    """Validate a request to roll low-level events up into a semantic summary.

    At least one selector (``video_id`` or ``trace_id``) must be supplied.
    """

    video_id = serializers.IntegerField(required=False, allow_null=True)
    trace_id = serializers.CharField(required=False, allow_blank=True)
    analysis_id = serializers.IntegerField(required=False, allow_null=True)
    scope = serializers.CharField(required=False, default="video")

    def validate(self, attrs):
        if attrs.get("video_id") is None and not attrs.get("trace_id"):
            raise serializers.ValidationError(
                "Provide at least one of 'video_id' or 'trace_id'."
            )
        return attrs
