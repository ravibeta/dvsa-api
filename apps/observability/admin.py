"""Admin registration for commentary events."""

from django.contrib import admin

from .models import CommentaryEventRecord


@admin.register(CommentaryEventRecord)
class CommentaryEventRecordAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "source", "video_id", "frame_index", "commentary")
    list_filter = ("source", "schema_version")
    search_fields = ("trace_id", "correlation_key", "commentary", "event_id")
    readonly_fields = ("created_at",)
    ordering = ("-timestamp",)
