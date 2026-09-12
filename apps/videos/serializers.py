"""Video serializers."""

from rest_framework import serializers
from .models import Video, VideoEntity

class VideoSerializer(serializers.ModelSerializer):
    """Serializer for video model."""
    user_email = serializers.CharField(source='user.email', read_only=True)
    
    class Meta:
        model = Video
        fields = [
            'id', 'user_email', 'title', 'description', 'file', 'thumbnail',
            'duration', 'file_size', 'resolution', 'frame_rate',
            'latitude', 'longitude', 'altitude', 'location_name',
            'status', 'is_public', 'tags', 'metadata',
            'created_at', 'updated_at', 'processed_at'
        ]
        read_only_fields = ['id', 'user_email', 'created_at', 'updated_at', 'processed_at']
    
    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)


class VideoEntitySerializer(serializers.ModelSerializer):
    """Serializer for the account-scoped VideoEntity (ported pipeline)."""

    class Meta:
        model = VideoEntity
        fields = '__all__'


class FrameExtractRequestSerializer(serializers.Serializer):
    """Request body for ``POST /api/v1/videos/extract-frames/``."""

    account_id = serializers.CharField()
    video_sas_url = serializers.CharField(required=False, allow_blank=True)
    sas_url = serializers.CharField(required=False, allow_blank=True)
    video_id = serializers.IntegerField(required=False)
    stride = serializers.FloatField(required=False, default=10.0, min_value=0.1)

    def resolved_sas_url(self):
        data = self.validated_data
        return data.get("video_sas_url") or data.get("sas_url") or None


class FrameResultSerializer(serializers.Serializer):
    """One extracted frame in the paginated response."""

    frame_number = serializers.IntegerField()
    blob_name = serializers.CharField()
    sas_url = serializers.CharField()
    t = serializers.FloatField(required=False, allow_null=True)
