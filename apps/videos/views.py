"""Video views."""

import datetime
import logging

from rest_framework import generics, permissions, status, viewsets
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from core.azure import AzureEnvironmentConfig, create_session_azure_environment
from core.permissions import IsOwnerOrReadOnly

from .models import Video, VideoEntity
from .serializers import VideoEntitySerializer, VideoSerializer

logger = logging.getLogger("apps.videos")

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


# ==========================================================================
# Account-scoped pipeline (ported from ezvision my_droneworld_api/videos)
# ==========================================================================
class VideoEntityViewSet(viewsets.ModelViewSet):
    """CRUD for account-scoped VideoEntity rows."""

    queryset = VideoEntity.objects.all()
    serializer_class = VideoEntitySerializer
    permission_classes = [permissions.IsAuthenticated]

    def list(self, request, *args, **kwargs):
        account_id = request.query_params.get("account_id")
        qs = VideoEntity.objects.filter(account_id=account_id) if account_id \
            else VideoEntity.objects.none()
        return Response(self.get_serializer(qs, many=True).data, status=status.HTTP_200_OK)

    def create(self, request, *args, **kwargs):
        account_id = request.data.get("account_id")
        video = VideoEntity()
        video.create_video(account_id=account_id)
        return Response(self.get_serializer(video).data, status=status.HTTP_201_CREATED)


class VideoUploadAPIView(APIView):
    """Upload a video to the ``sadronevideo`` account and register a VideoEntity.

    Saving the VideoEntity fires the ``post_save`` signal, which runs the Azure
    ingestion/indexing pipeline (see ``apps/videos/signals.py``).
    """

    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, format=None):
        file_obj = request.FILES.get("file")
        account_id = request.data.get("account_id")
        if not file_obj or not account_id:
            return Response({"error": "file and account_id are required"},
                            status=status.HTTP_400_BAD_REQUEST)
        cfg = AzureEnvironmentConfig.from_settings()
        blob_name = f"{account_id}/{file_obj.name}"
        try:
            from azure.storage.blob import BlobSasPermissions, generate_blob_sas

            from core.azure.blob import service_client

            svc = service_client(cfg)
            svc.get_blob_client(container=cfg.input_container, blob=blob_name).upload_blob(
                file_obj, overwrite=True
            )
            sas_token = generate_blob_sas(
                account_name=cfg.storage_account, container_name=cfg.input_container,
                blob_name=blob_name, account_key=cfg.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            )
            sas_url = (f"https://{cfg.storage_account}.blob.core.windows.net/"
                       f"{cfg.input_container}/{blob_name}?{sas_token}")
            video = VideoEntity()
            video.create_video(account_id=account_id, sas_url=sas_url)
            return Response(VideoEntitySerializer(video).data, status=status.HTTP_201_CREATED)
        except Exception as exc:  # noqa: BLE001
            logger.exception("video upload failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatAPIView(APIView):
    """Answer a question over an account's indexed frames (agentic synthesis)."""

    parser_classes = [MultiPartParser, FormParser, JSONParser]
    permission_classes = [permissions.IsAuthenticated]

    def put(self, request, pk=None, format=None):
        account_id = request.data.get("account_id")
        query_text = request.data.get("query")
        if not account_id or not query_text:
            return Response({"error": "query and account_id are required"},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            env = create_session_azure_environment(
                f"account-{account_id}", user_id=request.user.pk
            )
            answer = env.ask(query_text, str(account_id))
            return Response({"text": answer, "imageUrl": None, "downloadUrl": None},
                            status=status.HTTP_200_OK)
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)