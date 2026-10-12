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

        # DVSA_QWEN_ENABLED is read once, here at startup: when on, this API
        # process owns a local Qwen (Ollama) server for its whole lifetime.
        from core.azure import local_qwen_server

        if local_qwen_server.should_autostart_for_this_process():
            local_qwen_server.start_if_enabled()
