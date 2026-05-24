"""Video admin configuration."""

from django.contrib import admin
from .models import Video

@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'status', 'is_public', 'created_at')
    list_filter = ('status', 'is_public', 'created_at')
    search_fields = ('title', 'description', 'user__email')
    readonly_fields = ('created_at', 'updated_at', 'processed_at')