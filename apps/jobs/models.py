"""Analytics jobs run asynchronously (Celery) on uploaded :class:`apps.media.models.MediaAsset`s."""

import uuid

from django.conf import settings
from django.db import models
from django.db.models import F, Value
from django.db.models.functions import Coalesce
from django.utils import timezone


class AnalyticsJob(models.Model):
    class JobType(models.TextChoices):
        VEHICLE_TRACKING = "VEHICLE_TRACKING", "Vehicle tracking"
        GEOSPATIAL_MAPPING = "GEOSPATIAL_MAPPING", "Geospatial mapping (orthomosaic)"
        EXTRACT_FRAMES = "EXTRACT_FRAMES", "Extract frames"
        # Reserved: the full video indexer run followed by its frame extractions, object
        # detections and the other indexer stages. Costly, so never a bulk job: it must be
        # requested per video and is rejected by this API (see services.validate_job_type).
        FULL_PIPELINE = "FULL_PIPELINE", "Full pipeline (per video only)"

    class Status(models.TextChoices):
        QUEUED = "QUEUED", "Queued"
        PROCESSING = "PROCESSING", "Processing"
        COMPLETED = "COMPLETED", "Completed"
        FAILED = "FAILED", "Failed"
        CANCELLED = "CANCELLED", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              related_name="analytics_jobs")
    media_asset = models.ForeignKey("media.MediaAsset", on_delete=models.CASCADE,
                                    related_name="jobs")
    celery_task_id = models.CharField(max_length=64, blank=True, default="")
    attempt = models.PositiveIntegerField(default=0)
    slot_waits = models.PositiveIntegerField(default=0)
    job_type = models.CharField(max_length=32, choices=JobType.choices)
    model_name = models.CharField(max_length=255, blank=True, default="")
    model_version = models.CharField(max_length=64, blank=True, default="")
    parameters = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.QUEUED,
                              db_index=True)
    progress_percent = models.PositiveSmallIntegerField(default=0)
    stage = models.CharField(max_length=64, blank=True, default="")
    result_summary = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"AnalyticsJob({self.id}, {self.job_type}, {self.status})"


S = AnalyticsJob.Status
# Legal status edges. PROCESSING -> QUEUED is internal (waiting for a busy pipeline slot);
# FAILED/CANCELLED -> QUEUED is the explicit retry.
TRANSITIONS = {
    S.QUEUED: {S.PROCESSING, S.CANCELLED},
    S.PROCESSING: {S.COMPLETED, S.FAILED, S.CANCELLED, S.QUEUED},
    S.FAILED: {S.QUEUED},
    S.CANCELLED: {S.QUEUED},
    S.COMPLETED: set(),
}
TERMINAL = {S.COMPLETED, S.FAILED, S.CANCELLED}


def transition(job_id, to: str, *, from_=None, **fields) -> bool:
    """Atomically move a job to ``to`` if its current status allows it.

    A conditional ``UPDATE ... WHERE status IN (...)``, so a worker finishing and a user
    cancelling can never both win. ``from_`` narrows the allowed source statuses.
    """
    sources = {s for s, targets in TRANSITIONS.items() if to in targets}
    if from_ is not None:
        sources &= set(from_)
    if not sources:
        return False
    now = timezone.now()
    if to == S.PROCESSING:
        fields.setdefault("started_at", Coalesce(F("started_at"), Value(now)))
    if to in TERMINAL:
        fields.setdefault("completed_at", now)
    return AnalyticsJob.objects.filter(pk=job_id, status__in=sources).update(
        status=to, updated_at=now, **fields) == 1


def set_progress(job_id, percent: int, stage: str = None) -> bool:
    """Monotonic progress update; ignored unless the job is still PROCESSING."""
    percent = max(0, min(100, int(percent)))
    fields = {"progress_percent": percent, "updated_at": timezone.now()}
    if stage is not None:
        fields["stage"] = stage[:64]
    return AnalyticsJob.objects.filter(pk=job_id, status=S.PROCESSING,
                                       progress_percent__lte=percent).update(**fields) == 1
