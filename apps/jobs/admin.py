"""Jobs admin configuration."""

from django.contrib import admin

from .models import AnalyticsJob


@admin.register(AnalyticsJob)
class AnalyticsJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'owner', 'job_type', 'status', 'progress_percent', 'attempt', 'updated_at')
    list_filter = ('job_type', 'status', 'created_at')
    search_fields = ('id', 'owner__email', 'celery_task_id')
    readonly_fields = ('created_at', 'updated_at', 'started_at', 'completed_at')
