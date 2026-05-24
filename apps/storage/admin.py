"""Storage admin configuration."""

from django.contrib import admin
from .models import StorageConfiguration

@admin.register(StorageConfiguration)
class StorageConfigurationAdmin(admin.ModelAdmin):
    list_display = ('storage_type', 'is_active', 'created_at')
    list_filter = ('storage_type', 'is_active', 'created_at')