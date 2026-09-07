"""Video views."""

import datetime
import logging
import os

from rest_framework import generics, permissions, status, viewsets
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from core.azure import AzureEnvironmentConfig, create_session_azure_environment
from core.azure import synth_screen
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

        # Refuse footage that is not camera-based. Only clear cases are rejected
        # (the screen is precision-tuned and defaults to "camera"); if screening
        # cannot run, the upload proceeds (fail-open).
        verdict = self._screen_uploaded_video(file_obj)
        if verdict in synth_screen.FABRICATED_VERDICTS:
            logger.info("rejecting upload %s: screened as %s", file_obj.name, verdict)
            return Response(
                {"error": "This video appears to be AI-generated or fabricated and "
                          "cannot be uploaded. Only camera-captured drone footage is "
                          "accepted.",
                 "verdict": verdict},
                status=status.HTTP_400_BAD_REQUEST,
            )

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

    @staticmethod
    def _screen_uploaded_video(file_obj):
        """Return the fabrication-screen verdict for an uploaded clip.

        Writes the upload to a temp file, runs :func:`core.azure.synth_screen.screen`,
        then removes the temp file and rewinds ``file_obj`` for the blob upload.
        Fail-open: any error (no ffprobe/cv2, unreadable clip) returns ``"camera"``
        so a missing video toolchain never blocks a legitimate upload.
        """
        import tempfile  # noqa: PLC0415

        suffix = os.path.splitext(file_obj.name or "")[1] or ".mp4"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                for chunk in file_obj.chunks():
                    fh.write(chunk)
            return synth_screen.screen(tmp_path).get("verdict", "camera")
        except Exception:  # noqa: BLE001
            logger.exception("video screening failed; allowing upload (fail-open)")
            return "camera"
        finally:
            file_obj.seek(0)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as exc:
                    logger.info("screen temp cleanup failed: %s", exc)


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


class BaselineTestAPIView(APIView):
    """Raw ``qwen2.5vl:7b`` (Ollama) answer — a baseline peer of ``chat/``.

    Mirrors :class:`ChatAPIView`'s request (``account_id`` + ``query``, plus an
    optional ``image`` upload) and response shape, but bypasses all DVSA
    agentic/RAG synthesis: the question (and image) go straight to a locally
    hosted ``qwen2.5vl:7b`` via Ollama and only that model's reply is returned.
    This gives a direct, side-by-side baseline for the regular chat endpoint.
    """

    parser_classes = [MultiPartParser, FormParser, JSONParser]
    permission_classes = [permissions.IsAuthenticated]

    def put(self, request, pk=None, format=None):
        account_id = request.data.get("account_id")
        query_text = request.data.get("query")
        if not account_id or not query_text:
            return Response({"error": "query and account_id are required"},
                            status=status.HTTP_400_BAD_REQUEST)
        image_file = request.FILES.get("image")
        image_bytes = image_file.read() if image_file else None
        try:
            from core.azure.qwen_ollama import run_ollama_qwen

            cfg = AzureEnvironmentConfig.from_settings()
            answer = run_ollama_qwen(
                cfg.ollama_host, cfg.ollama_qwen_model, query_text, image_bytes
            )
            return Response({"text": answer, "imageUrl": None, "downloadUrl": None},
                            status=status.HTTP_200_OK)
        except Exception as exc:  # noqa: BLE001
            logger.exception("baseline-test failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CornersAPIView(APIView):
    """Extract the four survey-area corner frames from an account's video.

    A direct tool endpoint that bypasses the query / RAG / agentic path: given
    an account's uploaded video, it runs the corner-extraction routine (see
    :mod:`core.azure.survey_corners`), writes the four frames to a ``corners``
    folder alongside the video's frames in blob storage
    (``{account_id}/images/{video_pk}/corners/{label}.jpg``, overwriting any
    existing ones), and returns a downloadable read-SAS URL for each corner,
    ordered bottom-left -> top-left -> top-right -> bottom-right.

    Request (``IsAuthenticated``): ``account_id`` (required), plus optional
    ``video_id`` (a :class:`VideoEntity` pk) or ``sas_url`` to name the source
    video. Without either, the account's most recent ``VideoEntity`` is used;
    ``video_pk`` defaults to ``0`` when no entity can be resolved.
    """

    parser_classes = [MultiPartParser, FormParser, JSONParser]
    permission_classes = [permissions.IsAuthenticated]

    def put(self, request, pk=None, format=None):
        account_id = request.data.get("account_id")
        if not account_id:
            return Response({"error": "account_id is required"},
                            status=status.HTTP_400_BAD_REQUEST)
        video_id = request.data.get("video_id")
        sas_url = request.data.get("sas_url")

        entity = None
        if video_id:
            entity = VideoEntity.objects.filter(pk=video_id,
                                                account_id=account_id).first()
            if entity is None:
                return Response({"error": "video not found"},
                                status=status.HTTP_404_NOT_FOUND)
        elif not sas_url:
            entity = VideoEntity.objects.filter(
                account_id=account_id).order_by("-created").first()

        source_url = sas_url or (entity.sas_url if entity else None)
        if not source_url:
            return Response({"error": "no video found for account"},
                            status=status.HTTP_404_NOT_FOUND)
        video_pk = entity.pk if entity else 0

        try:
            from core.azure import blob as azure_blob
            from core.azure.survey_corners import extract_corner_frames

            cfg = AzureEnvironmentConfig.from_settings()
            video_path = azure_blob.download_blob_to_temp(source_url)
            try:
                frames = extract_corner_frames(video_path)
            finally:
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except OSError as exc:
                        logger.info("corners temp cleanup failed: %s", exc)

            corners = []
            for label, jpg_bytes, meta in frames:
                blob_name = f"{account_id}/images/{video_pk}/corners/{label}.jpg"
                download_url = azure_blob.put_blob_and_sas(cfg, blob_name, jpg_bytes)
                corners.append({"label": label, "downloadUrl": download_url, **meta})
            return Response({"account_id": str(account_id), "video_pk": video_pk,
                             "count": len(corners), "corners": corners},
                            status=status.HTTP_200_OK)
        except Exception as exc:  # noqa: BLE001
            logger.exception("corners extraction failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
