"""Celery task that runs one analytics job (``process_drone_video_job``)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task
from django.db.models import F

from apps.notifications import events
from apps.notifications.broadcast import broadcast_live_alert

from . import runners
from .cancel import JobCancelled, JobCancelToken
from .models import AnalyticsJob, set_progress, transition
from .services import dispatch, job_event_data

logger = logging.getLogger("apps.jobs")

S = AnalyticsJob.Status
SLOT_WAIT_S = 60
MAX_SLOT_WAITS = 10


def _alert_data(job: AnalyticsJob, summary: Dict[str, Any]) -> Dict[str, Any]:
    features = (summary.get("geojson") or {}).get("features") or []
    confidences = [f["properties"].get("confidence") for f in features
                   if f["properties"].get("confidence") is not None]
    first = [f["properties"].get("first_t") for f in features
             if f["properties"].get("first_t") is not None]
    return {**job_event_data(job), "count": len(features), "geojson": summary.get("geojson"),
            "confidence": max(confidences) if confidences else None,
            "frame_timestamp": min(first) if first else None}


@shared_task(name="jobs.process_drone_video_job", acks_late=True)
def process_drone_video_job(job_id: str) -> Dict[str, Any]:
    """QUEUED -> PROCESSING -> COMPLETED | FAILED | CANCELLED. Never raises."""
    if not transition(job_id, S.PROCESSING, from_=(S.QUEUED,), stage="starting",
                      progress_percent=1):
        # Cancelled before it started, or another worker already owns it.
        return {"job_id": str(job_id), "skipped": True}
    job = AnalyticsJob.objects.select_related("media_asset").get(pk=job_id)

    def progress(percent: int, stage: str) -> None:
        set_progress(job.pk, percent, stage)

    try:
        from core.storage import get_object_store  # noqa: PLC0415

        ctx = runners.JobContext(job=job, asset=job.media_asset, store=get_object_store(),
                                 cancel=JobCancelToken(job.pk), progress=progress)
        summary = runners.run(ctx)
    except runners.SlotBusy as exc:
        if job.slot_waits < MAX_SLOT_WAITS and transition(
                job.pk, S.QUEUED, from_=(S.PROCESSING,), stage="waiting for slot",
                slot_waits=F("slot_waits") + 1):
            dispatch(job, countdown=SLOT_WAIT_S, new_attempt=False)
            return {"job_id": str(job.pk), "status": S.QUEUED, "waiting": True}
        return _fail(job, f"pipeline busy: {exc}")
    except JobCancelled:
        return {"job_id": str(job.pk), "status": S.CANCELLED}
    except Exception as exc:  # noqa: BLE001 - recorded on the job, never raised into Celery
        logger.warning("job %s (%s) failed: %s", job.pk, job.job_type, exc,
                       exc_info=not isinstance(exc, runners.JobFailed))
        return _fail(job, str(exc))

    if not transition(job.pk, S.COMPLETED, from_=(S.PROCESSING,), progress_percent=100,
                      stage="completed", result_summary=summary):
        return {"job_id": str(job.pk), "status": S.CANCELLED}  # cancelled while finishing
    job.refresh_from_db()
    broadcast_live_alert(job.owner_id, events.JOB_COMPLETED,
                         {**job_event_data(job), "summary": summary.get("summary")})
    features = (summary.get("geojson") or {}).get("features") or []
    if job.job_type == AnalyticsJob.JobType.VEHICLE_TRACKING and features \
            and len(features) >= runners.min_vehicles_for_alert():
        broadcast_live_alert(job.owner_id, events.DETECTION_TRIGGERED, _alert_data(job, summary))
    return {"job_id": str(job.pk), "status": S.COMPLETED}


def _fail(job: AnalyticsJob, reason: str) -> Dict[str, Any]:
    if transition(job.pk, S.FAILED, from_=(S.PROCESSING,), stage="failed",
                  error_message=reason[:500]):
        job.refresh_from_db()
        broadcast_live_alert(job.owner_id, events.JOB_FAILED,
                             {**job_event_data(job), "error": job.error_message})
        return {"job_id": str(job.pk), "status": S.FAILED}
    return {"job_id": str(job.pk), "status": S.CANCELLED}
