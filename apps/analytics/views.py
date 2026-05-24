"""Analytics views."""

from rest_framework import generics, permissions
from .models import Analysis
from .serializers import AnalysisSerializer

class AnalysisListView(generics.ListAPIView):
    """List all analyses for the current user."""
    serializer_class = AnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return Analysis.objects.filter(user=self.request.user)

class AnalysisDetailView(generics.RetrieveAPIView):
    """Retrieve analysis details."""
    serializer_class = AnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return Analysis.objects.filter(user=self.request.user)