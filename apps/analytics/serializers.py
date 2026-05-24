"""Analytics serializers."""

from rest_framework import serializers
from .models import Analysis

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