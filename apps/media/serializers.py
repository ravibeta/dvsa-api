"""Serializers for direct uploads."""

from rest_framework import serializers

from .models import FLIGHT_ID_PATTERN, MediaAsset


class UploadDestinationRequestSerializer(serializers.Serializer):
    filename = serializers.CharField(max_length=255)
    content_type = serializers.CharField(max_length=100)
    file_size_bytes = serializers.IntegerField(min_value=1)
    flight_id = serializers.RegexField(rf"^{FLIGHT_ID_PATTERN}$", required=False, allow_blank=True)
    metadata = serializers.DictField(required=False)
    checksum_sha256 = serializers.RegexField(r"^[A-Fa-f0-9]{64}$", required=False, allow_blank=True)
    account_id = serializers.CharField(max_length=64, required=False)


class CompleteRequestSerializer(serializers.Serializer):
    job_type = serializers.CharField(max_length=32, required=False)
    parameters = serializers.DictField(required=False)
    account_id = serializers.CharField(max_length=64, required=False)


class MediaAssetSerializer(serializers.ModelSerializer):
    media_id = serializers.UUIDField(source="id", read_only=True)
    blob_path = serializers.CharField(source="object_key", read_only=True)

    class Meta:
        model = MediaAsset
        fields = ("media_id", "flight_id", "filename", "content_type", "file_size_bytes",
                  "storage_provider", "bucket", "object_key", "blob_path", "upload_status",
                  "etag", "checksum_sha256", "metadata", "error_message", "created_at",
                  "updated_at", "uploaded_at")
        read_only_fields = fields
