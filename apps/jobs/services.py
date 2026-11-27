"""Create, dispatch, cancel and retry analytics jobs (shared by the media and jobs APIs)."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from django.conf import settings
from django.db.models import F
from django.utils import timezone

from .models import AnalyticsJob, transition

logger = logging.getLogger("apps.jobs")

JT, S = AnalyticsJob.JobType, AnalyticsJob.Status
BULK_JOB_TYPES = (JT.VEHICLE_TRACKING, JT.GEOSPATIAL_MAPPING, JT.EXTRACT_FRAMES)
DEFAULT_JOB_TYPE = JT.EXTRACT_FRAMES
FULL_PIPELINE_MESSAGE = ("FULL_PIPELINE runs the full video indexer workflow and is not supported "
                         "for bulk jobs; request it per video")


class UnsupportedJobType(ValueError):
    pass


def validate_job_type(job_type: Optional[str]) -> str:
    job_type = (job_type or DEFAULT_JOB_TYPE).strip().upper()
    if job_type == JT.FULL_PIPELINE:
        raise UnsupportedJobType(FULL_PIPELINE_MESSAGE)
    if job_type not in BULK_JOB_TYPES:
        raise UnsupportedJobType(f"unknown job_type {job_type!r}; use one of {', '.join(BULK_JOB_TYPES)}")
    return job_type


def default_model_name(job_type: str) -> str:
    if job_type == JT.VEHICLE_TRACKING:
        return str(getattr(settings, "DVSA_VEHICLES_WEIGHTS", "yolo26x.pt"))
    if job_type == JT.GEOSPATIAL_MAPPING:
        return "opendronemap"
    return "frame-extract"


def create_job(owner, asset, job_type: Optional[str] = None, *, model_name: str = "",
               model_version: str = "", parameters: Optional[Dict[str, Any]] = None,
               dispatch_now: bool = True) -> AnalyticsJob:
    job_type = validate_job_type(job_type)
    job = AnalyticsJob.objects.create(
        owner=owner, media_asset=asset, job_type=job_type,
        model_name=model_name or default_model_name(job_type), model_version=model_version or "",
        parameters=parameters or {})
    return dispatch(job) if dispatch_now else job


def dispatch(job: AnalyticsJob, *, countdown: Optional[int] = None,
             new_attempt: bool = True) -> AnalyticsJob:
    """Queue ``process_drone_video_job`` under a fresh Celery task id.

    Every dispatch gets its own task id, and ``attempt`` counts user-visible attempts
    (first run and each retry), so retries and recovery never reuse an id.
    """
    from .tasks import process_drone_video_job  # noqa: PLC0415

    task_id = str(uuid.uuid4())
    fields: Dict[str, Any] = {"celery_task_id": task_id, "updated_at": timezone.now()}
    if new_attempt:
        fields["attempt"] = F("attempt") + 1
    AnalyticsJob.objects.filter(pk=job.pk).update(**fields)
    try:
        process_drone_video_job.apply_async((str(job.pk),), task_id=task_id, countdown=countdown)
    except Exception as exc:  # noqa: BLE001 - e.g. broker unreachable
        logger.warning("could not queue job %s: %s", job.pk, exc)
        transition(job.pk, S.FAILED, from_=(S.QUEUED,), error_message=f"could not queue job: {exc}"[:500])
    job.refresh_from_db()
    return job


def _revoke(task_id: str) -> None:
    """Drop a not-yet-started task from the queue (never kills a running one)."""
    if not task_id or getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False):
        return
    try:
        from config.celery import app  # noqa: PLC0415

        app.control.revoke(task_id)
    except Exception as exc:  # noqa: BLE001 - cancellation is cooperative anyway
        logger.info("revoke of task %s failed: %s", task_id, exc)


def cancel_job(job: AnalyticsJob) -> bool:
    """Mark the job CANCELLED; a running task stops at its next stage boundary."""
    if not transition(job.pk, S.CANCELLED, from_=(S.QUEUED, S.PROCESSING), stage="cancelled"):
        return False
    _revoke(job.celery_task_id)
    from apps.notifications.broadcast import broadcast_live_alert  # noqa: PLC0415
    from apps.notifications.events import JOB_CANCELLED  # noqa: PLC0415

    job.refresh_from_db()
    broadcast_live_alert(job.owner_id, JOB_CANCELLED, job_event_data(job))
    return True


def retry_job(job: AnalyticsJob) -> Optional[AnalyticsJob]:
    """Re-queue a FAILED or CANCELLED job as a new attempt (same job id)."""
    if not transition(job.pk, S.QUEUED, from_=(S.FAILED, S.CANCELLED), progress_percent=0,
                      stage="", error_message="", slot_waits=0, started_at=None, completed_at=None):
        return None
    return dispatch(job)


def job_event_data(job: AnalyticsJob) -> Dict[str, Any]:
    return {"job_id": str(job.id), "media_id": str(job.media_asset_id), "job_type": job.job_type,
            "status": job.status, "progress_percent": job.progress_percent,
            "flight_id": getattr(job.media_asset, "flight_id", "")}
