"""App config for the storage app."""

from django.apps import AppConfig


class StorageConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.storage"
    label = "storage"
    verbose_name = "Storage"
