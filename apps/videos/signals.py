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
