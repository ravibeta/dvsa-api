"""Analytics admin configuration."""

from django.contrib import admin
from .models import Analysis

@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ('video', 'user', 'status', 'objects_detected', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('video__title', 'user__email')
    readonly_fields = ('created_at', 'updated_at', 'completed_at')