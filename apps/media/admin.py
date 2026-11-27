"""Media admin configuration."""

from django.contrib import admin

from .models import MediaAsset


@admin.register(MediaAsset)
class MediaAssetAdmin(admin.ModelAdmin):
    list_display = ('id', 'owner', 'filename', 'upload_status', 'file_size_bytes', 'created_at')
    list_filter = ('upload_status', 'storage_provider', 'created_at')
    search_fields = ('id', 'filename', 'object_key', 'owner__email', 'flight_id')
    readonly_fields = ('created_at', 'updated_at', 'uploaded_at')
