"""Media uploaded directly to cloud object storage (two-step presigned upload)."""

import os
import uuid

from django.conf import settings
from django.db import models

FLIGHT_ID_PATTERN = r"[A-Za-z0-9_.-]{1,64}"
VIDEO_EXTENSIONS = (".mp4", ".mov")


class MediaAsset(models.Model):
    """One uploaded file. Its object key embeds the asset id, so keys never collide
    across retries, re-uploads of the same file, or recovery runs."""

    class UploadStatus(models.TextChoices):
        PENDING = "PENDING", "Pending"
        UPLOADED = "UPLOADED", "Uploaded"
        FAILED = "FAILED", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              related_name="media_assets")
    flight_id = models.CharField(max_length=64, blank=True, default="", db_index=True)
    filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100)
    file_size_bytes = models.BigIntegerField()
    storage_provider = models.CharField(max_length=20)
    bucket = models.CharField(max_length=255)
    object_key = models.CharField(max_length=1024, unique=True)
    upload_status = models.CharField(max_length=10, choices=UploadStatus.choices,
                                     default=UploadStatus.PENDING, db_index=True)
    etag = models.CharField(max_length=128, blank=True, default="")
    checksum_sha256 = models.CharField(max_length=64, blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    idempotency_key = models.CharField(max_length=128, blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    uploaded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["owner", "idempotency_key"],
                                    condition=~models.Q(idempotency_key=""),
                                    name="media_unique_owner_idempotency_key"),
        ]

    def __str__(self):
        return f"MediaAsset({self.id}, {self.filename}, {self.upload_status})"

    @property
    def extension(self) -> str:
        return os.path.splitext(self.filename)[1].lower()

    @property
    def is_video(self) -> bool:
        return self.extension in VIDEO_EXTENSIONS
