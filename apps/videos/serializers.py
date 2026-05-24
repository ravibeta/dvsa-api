"""Video serializers."""

from rest_framework import serializers
from .models import Video

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