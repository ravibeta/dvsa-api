"""Storage views."""

from rest_framework import viewsets, permissions
from .models import StorageConfiguration

class StorageViewSet(viewsets.ReadOnlyModelViewSet):
    """Storage configuration viewset."""
    queryset = StorageConfiguration.objects.all()
    permission_classes = [permissions.IsAdminUser]