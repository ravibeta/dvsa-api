"""Analytics models."""

from django.db import models
from django.contrib.auth import get_user_model
from apps.videos.models import Video

User = get_user_model()

class Analysis(models.Model):
    """Video analysis results."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    video = models.OneToOneField(Video, on_delete=models.CASCADE, related_name='analysis')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Correlation id shared by all commentary events emitted for this run
    # (see apps.observability). Lets run_semantic_agent(trace_id=...) and OTel
    # traces be reached straight from the Analysis row. Blank until commentary
    # runs (COMMENTARY_ENABLED).
    trace_id = models.CharField(max_length=32, blank=True, default="", db_index=True)

    objects_detected = models.IntegerField(default=0)
    scenes_detected = models.IntegerField(default=0)
    anomalies_found = models.IntegerField(default=0)
    
    results = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']