"""Video views."""

from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Video
from .serializers import VideoSerializer
from core.permissions import IsOwnerOrReadOnly

class VideoListCreateView(generics.ListCreateAPIView):
    """List all videos or create a new video."""
    serializer_class = VideoSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class VideoRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update or delete a video."""
    serializer_class = VideoSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    
    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)

class VideoProcessView(APIView):
    """Trigger video processing."""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, pk):
        try:
            video = Video.objects.get(pk=pk, user=request.user)
            video.status = 'processing'
            video.save()
            return Response(
                {'message': 'Video processing started', 'status': video.status},
                status=status.HTTP_200_OK
            )
        except Video.DoesNotExist:
            return Response({'error': 'Video not found'}, status=status.HTTP_404_NOT_FOUND)