"""Video models."""

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