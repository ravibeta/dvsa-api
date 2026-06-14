"""Persistence for commentary events — one wide row per event.

``CommentaryEventRecord`` is intentionally a *flat, wide* table: scalar columns
for everything queryable plus three JSON columns (``attributes`` / ``metrics`` /
``metadata``) that absorb open-ended payload. Nothing is pre-aggregated — roll-ups
happen at query time (see :mod:`apps.observability.aggregation`).

The layer is kept **orthogonal** to the vision pipeline: ``video_id`` /
``analysis_id`` are plain integers, not foreign keys, so external systems and
agents can inject commentary referencing ids the observability app does not own.
"""

from django.db import models


class CommentaryEventRecord(models.Model):
    """A single stored commentary/observability event."""

    # Identity & correlation
    event_id = models.CharField(max_length=32, unique=True, db_index=True)
    trace_id = models.CharField(max_length=32, db_index=True)
    span_id = models.CharField(max_length=16)
    parent_span_id = models.CharField(max_length=16, blank=True, null=True)
    correlation_key = models.CharField(max_length=255, blank=True, default="", db_index=True)

    # Temporal / source context
    timestamp = models.DateTimeField(db_index=True)
    source = models.CharField(max_length=128, db_index=True)
    video_id = models.IntegerField(null=True, blank=True, db_index=True)
    analysis_id = models.IntegerField(null=True, blank=True, db_index=True)
    frame_index = models.IntegerField(null=True, blank=True)
    segment_start = models.FloatField(null=True, blank=True)
    segment_end = models.FloatField(null=True, blank=True)

    # Commentary payload
    commentary = models.TextField(blank=True, default="")
    attributes = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    schema_version = models.CharField(max_length=16, default="1.0")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["trace_id", "frame_index"]),
            models.Index(fields=["video_id", "frame_index"]),
            models.Index(fields=["source", "timestamp"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - repr convenience
        return "%s @ frame %s: %s" % (self.source, self.frame_index, self.commentary[:60])
