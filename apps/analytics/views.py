"""Analytics views."""

from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.videos.models import Video
from .models import Analysis
from .routines import available_routines
from .serializers import AnalysisSerializer, RunAnalysisSerializer
from .tasks import run_video_analysis


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


class RoutineListView(APIView):
    """List the available vision routines and their metadata."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response({"routines": available_routines()})


class RunAnalysisView(APIView):
    """Queue a vision-routine analysis run for one of the user's videos."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, video_id):
        try:
            video = Video.objects.get(pk=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response(
                {"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND
            )

        serializer = RunAnalysisSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        analysis, _ = Analysis.objects.get_or_create(
            video=video, defaults={"user": request.user}
        )
        analysis.user = request.user
        analysis.status = "pending"
        analysis.error_message = None
        analysis.save()

        run_video_analysis.delay(
            analysis.id,
            data["routines"],
            params=data.get("params", {}),
            frame_step=data["frame_step"],
            max_frames=data["max_frames"],
        )

        return Response(
            AnalysisSerializer(analysis).data, status=status.HTTP_202_ACCEPTED
        )
