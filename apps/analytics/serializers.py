"""Analytics serializers."""

from rest_framework import serializers
from .models import Analysis
from .routines import available_routines, get_routine

class AnalysisSerializer(serializers.ModelSerializer):
    """Serializer for analysis model."""
    video_title = serializers.CharField(source='video.title', read_only=True)

    class Meta:
        model = Analysis
        fields = [
            'id', 'video_title', 'status', 'objects_detected',
            'scenes_detected', 'anomalies_found', 'results',
            'created_at', 'updated_at', 'completed_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'completed_at']


class RunAnalysisSerializer(serializers.Serializer):
    """Validate a request to run vision routines against a video."""

    routines = serializers.ListField(
        child=serializers.CharField(), allow_empty=False,
        help_text="Routine names to run (see the routines listing endpoint)."
    )
    params = serializers.DictField(
        required=False, default=dict,
        help_text="Optional per-routine parameters, keyed by routine name."
    )
    frame_step = serializers.IntegerField(required=False, default=30, min_value=1)
    max_frames = serializers.IntegerField(required=False, default=300, min_value=1)

    def validate_routines(self, value):
        unknown = []
        for name in value:
            try:
                get_routine(name)
            except KeyError:
                unknown.append(name)
        if unknown:
            valid = [r["name"] for r in available_routines()]
            raise serializers.ValidationError(
                f"Unknown routine(s): {unknown}. Available: {valid}"
            )
        return value