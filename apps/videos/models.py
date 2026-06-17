"""Video models."""

from urllib.parse import urlparse

from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Video(models.Model):
    """Video model for drone footage."""
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('processed', 'Processed'),
        ('failed', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='videos')
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to='videos/')
    thumbnail = models.ImageField(upload_to='thumbnails/', blank=True, null=True)
    
    duration = models.IntegerField(blank=True, null=True)
    file_size = models.BigIntegerField(blank=True, null=True)
    resolution = models.CharField(max_length=20, blank=True, null=True)
    frame_rate = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    
    latitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)
    altitude = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    location_name = models.CharField(max_length=255, blank=True, null=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    is_public = models.BooleanField(default=False)
    
    tags = models.CharField(max_length=500, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return self.title


class VideoEntity(models.Model):
    """Account-scoped drone video (ported from ezvision my_droneworld_api).

    Distinct from :class:`Video` (the user-owned upload model): ``VideoEntity``
    is keyed on ``account_id`` and a blob ``sas_url`` and drives the Azure
    ingestion/indexing pipeline in :mod:`core.azure`. A ``post_save`` signal
    kicks off indexing (see ``apps/videos/signals.py``).
    """

    class Status(models.TextChoices):
        INITIALIZED = "Initialized", "Initialized"
        PROCESSING = "Processing", "Processing"
        COMPLETED = "Completed", "Completed"
        CANCELED = "Canceled", "Canceled"
        RESERVED = "Reserved", "Reserved"

    account_id = models.CharField(max_length=255)
    video_url = models.CharField(null=True, blank=True, max_length=1024)
    index_name = models.CharField(null=True, blank=True, max_length=255)
    sas_url = models.URLField(max_length=500)
    file_name = models.CharField(null=True, blank=True, max_length=255)
    status = models.CharField(max_length=20, choices=Status.choices,
                              default=Status.INITIALIZED)
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"VideoEntity(id={self.id}, account_id={self.account_id}, status={self.status})"

    def create_video(self, account_id, sas_url=None):
        self.account_id = account_id
        self.sas_url = sas_url
        self.file_name = self.get_name_from_url(sas_url)
        self.status = self.Status.INITIALIZED
        self.save()

    def update_video(self, **kwargs):
        for field_name, value in kwargs.items():
            if hasattr(self, field_name):
                setattr(self, field_name, value)
        self.status = self.Status.INITIALIZED
        self.save()

    def delete_video(self):
        self.delete()

    @staticmethod
    def get_name_from_url(sas_url):
        if not sas_url:
            return None
        return urlparse(sas_url).path.split("/")[-1]


class ImageEntity(models.Model):
    """A single extracted/derived frame belonging to a :class:`VideoEntity`."""

    video = models.ForeignKey(VideoEntity, related_name="images", on_delete=models.CASCADE)
    account_id = models.CharField(max_length=255)
    video_url = models.CharField(null=True, blank=True, max_length=1024)
    index_name = models.CharField(max_length=255)
    sas_url = models.CharField(max_length=1024)
    description = models.TextField(null=True, blank=True, max_length=4096)
    timestamp = models.TimeField(null=True, blank=True)
    location = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=255, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"ImageEntity(id={self.id}, video_id={self.video_id})"