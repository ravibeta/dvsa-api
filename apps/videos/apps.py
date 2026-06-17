"""App config for the videos app."""

from django.apps import AppConfig


class VideosConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.videos"
    label = "videos"
    verbose_name = "Videos"

    def ready(self):
        # Register post_save -> Azure ingestion pipeline.
        from . import signals  # noqa: F401
