"""Serializers for analytics jobs."""

from rest_framework import serializers

from .models import AnalyticsJob
from .services import UnsupportedJobType, validate_job_type


class AnalyticsJobSerializer(serializers.ModelSerializer):
    media_id = serializers.UUIDField(source="media_asset_id", read_only=True)

    class Meta:
        model = AnalyticsJob
        fields = ("id", "media_id", "job_type", "model_name", "model_version", "parameters",
                  "status", "progress_percent", "stage", "attempt", "celery_task_id",
                  "result_summary", "error_message", "created_at", "updated_at", "started_at",
                  "completed_at")
        read_only_fields = fields


class JobCreateSerializer(serializers.Serializer):
    media_id = serializers.UUIDField()
    job_type = serializers.CharField(max_length=32)
    model_name = serializers.CharField(max_length=255, required=False, allow_blank=True)
    model_version = serializers.CharField(max_length=64, required=False, allow_blank=True)
    parameters = serializers.DictField(required=False)

    def validate_job_type(self, value):
        try:
            return validate_job_type(value)
        except UnsupportedJobType as exc:
            raise serializers.ValidationError(str(exc)) from None
