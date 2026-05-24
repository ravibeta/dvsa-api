"""Storage models."""

from django.db import models

class StorageConfiguration(models.Model):
    """Cloud storage configuration."""
    STORAGE_TYPES = [
        ('azure', 'Azure Blob Storage'),
        ('aws', 'AWS S3'),
        ('local', 'Local Storage'),
    ]
    
    storage_type = models.CharField(max_length=20, choices=STORAGE_TYPES)
    is_active = models.BooleanField(default=True)
    config = models.JSONField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f'{self.get_storage_type_display()} Storage'