"""App config for the jobs app (asynchronous analytics jobs on uploaded media)."""

from django.apps import AppConfig


class JobsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.jobs"
    label = "jobs"
    verbose_name = "Analytics jobs"
