"""Signals for the videos app.

When a :class:`~apps.videos.models.VideoEntity` is created, kick off the Azure
ingestion/indexing pipeline (ported ``indexing_workflow``) via a session-scoped
Azure environment. Failures are logged, never raised, so a save always succeeds.
"""

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import VideoEntity

logger = logging.getLogger("apps.videos")


@receiver(post_save, sender=VideoEntity)
def video_entity_orthomosaic(sender, instance, created, **kwargs):
    """Start the orthomosaic in the background for a newly uploaded video.

    Registered before the ingestion receiver so it begins immediately instead of waiting for
    the (synchronous) indexing. It only spawns a daemon thread; any failure -- missing Docker,
    no ODM image, ODM error -- is recorded on the job and never reaches the upload.
    """
    if not created or not instance.sas_url:
        return
    try:
        from . import orthomosaic_jobs

        if orthomosaic_jobs.auto_enabled():
            orthomosaic_jobs.start_job(instance.account_id, instance.id, instance.sas_url)
    except Exception as exc:  # noqa: BLE001
        logger.info("Orthomosaic job for VideoEntity %s not started: %s", instance.id, exc)


@receiver(post_save, sender=VideoEntity)
def video_entity_post_save(sender, instance, created, **kwargs):
    if not created or not instance.sas_url:
        return
    try:
        from core.azure import create_session_azure_environment

        env = create_session_azure_environment(f"account-{instance.account_id}")
        result = env.ingest_video(instance.sas_url, instance.account_id, instance.id)
        logger.info("VideoEntity %s ingested: %s", instance.id, result)
    except Exception as exc:  # noqa: BLE001
        logger.info("Indexing VideoEntity %s failed: %s", instance.id, exc)
