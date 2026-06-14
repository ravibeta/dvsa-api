"""App config for the observability layer."""

from django.apps import AppConfig


class ObservabilityConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.observability"
    verbose_name = "DVSA Observability (Commentary)"
