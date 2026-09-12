"""Video views."""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid

from django.conf import settings
from django.core.exceptions import ValidationError
from rest_framework import generics, permissions, status, viewsets
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from core.azure import AzureEnvironmentConfig, create_session_azure_environment
from core.azure import synth_screen
from core.pagination import StandardResultsSetPagination
from core.permissions import IsOwnerOrReadOnly

from .models import ImageEntity, Video, VideoEntity
from .serializers import (FrameExtractRequestSerializer, FrameResultSerializer,
                          VideoEntitySerializer, VideoSerializer)

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


# --------------------------------------------------------------------------
# Hardened video-upload validation helpers.
#
# The original VideoUploadAPIView trusted the client-supplied filename and
# Content-Type, had no size limit, no real MP4 structural validation, no
# authorization check on account_id (IDOR), and leaked raw exception text
# to clients. The helpers below and the rewritten view fix each of these.
# See the constants/comments for what to tune for your deployment.
# --------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 500 * 1024 * 1024          # 500 MB hard cap
MAX_DURATION_SECONDS = 60 * 30                # 30 minutes
MAX_WIDTH = 7680                              # 8K ceiling
MAX_HEIGHT = 4320
MIN_WIDTH = 16
MIN_HEIGHT = 16

# MP4/ISO-BMFF "major brand" values accepted. Reject exotic/legacy
# container flavors even if they technically parse as ISO-BMFF.
ALLOWED_FTYP_BRANDS = {
    b"isom", b"iso2", b"mp41", b"mp42", b"avc1",
    b"M4V ", b"M4VH", b"M4VP", b"dash", b"mp71",
}

# Codec allowlist as reported by ffprobe's codec_name.
ALLOWED_VIDEO_CODECS = {"h264", "hevc", "h265"}
ALLOWED_AUDIO_CODECS = {"aac", "mp3", "ac3", "eac3", None}  # None = no audio track

# ffprobe/ffmpeg format_name values that count as "this is really MP4-ish".
ALLOWED_FFPROBE_FORMATS = {"mov,mp4,m4a,3gp,3g2,mj2"}

ACCOUNT_ID_RE = re.compile(r"^[0-9]{1,20}$")

FFPROBE_TIMEOUT_SECONDS = 15


class VideoValidationError(ValidationError):
    """Raised when an uploaded file fails MP4 validation."""


def _sniff_mp4_ftyp(path: str) -> None:
    """Verify the file starts with a well-formed ISO-BMFF ``ftyp`` box.

    Cheap first-pass check performed before handing the file to any
    heavier tool (ffprobe, the Azure pipeline). Rejects obvious non-MP4
    content (renamed images/executables/text files/etc.) without needing
    to fully parse the container.
    """
    with open(path, "rb") as fh:
        header = fh.read(12)

    if len(header) < 12:
        raise VideoValidationError("File is too small to be a valid MP4.")

    box_size = int.from_bytes(header[0:4], "big")
    box_type = header[4:8]
    brand = header[8:12]

    if box_type != b"ftyp":
        raise VideoValidationError(
            "File does not start with an MP4 'ftyp' box; not a valid MP4."
        )
    if box_size < 8:
        raise VideoValidationError("Malformed MP4 header (invalid box size).")
    if brand not in ALLOWED_FTYP_BRANDS:
        raise VideoValidationError(f"Unsupported MP4 major brand: {brand!r}.")


def _probe_with_ffprobe(path: str) -> dict:
    """Run ffprobe (argument list, no shell) and return parsed JSON.

    Raises VideoValidationError on any structural problem. Never invokes
    a shell and never interpolates ``path`` into a shell string, so there
    is no command-injection surface regardless of filename content.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VideoValidationError("Video inspection tool unavailable.") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoValidationError("Video inspection timed out.") from exc

    if result.returncode != 0:
        raise VideoValidationError("File could not be parsed as a valid video.")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoValidationError("Video metadata was unreadable.") from exc

    fmt = data.get("format", {})
    format_name = fmt.get("format_name", "")
    if not any(name in format_name for name in ALLOWED_FFPROBE_FORMATS):
        raise VideoValidationError(f"Unsupported container format: {format_name!r}.")

    try:
        duration = float(fmt.get("duration", "0"))
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise VideoValidationError("Video has no readable duration.")
    if duration > MAX_DURATION_SECONDS:
        raise VideoValidationError(
            f"Video exceeds maximum duration of {MAX_DURATION_SECONDS} seconds."
        )

    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise VideoValidationError("No video stream found in file.")

    for vs in video_streams:
        codec = vs.get("codec_name")
        if codec not in ALLOWED_VIDEO_CODECS:
            raise VideoValidationError(f"Unsupported video codec: {codec!r}.")
        width, height = vs.get("width", 0), vs.get("height", 0)
        if not (MIN_WIDTH <= width <= MAX_WIDTH) or not (MIN_HEIGHT <= height <= MAX_HEIGHT):
            raise VideoValidationError(
                f"Video resolution {width}x{height} outside allowed bounds."
            )

    for a_stream in audio_streams:
        codec = a_stream.get("codec_name")
        if codec not in ALLOWED_AUDIO_CODECS:
            raise VideoValidationError(f"Unsupported audio codec: {codec!r}.")

    return data


def _sanitize_display_name(original_name: str) -> str:
    """Produce a safe *display* filename. Never used for storage paths."""
    base = os.path.basename(original_name or "video.mp4")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base[:200] or "video.mp4"


def _validate_account_id(raw_account_id) -> int:
    if raw_account_id is None or not ACCOUNT_ID_RE.match(str(raw_account_id)):
        raise VideoValidationError("Invalid account_id.")
    return int(raw_account_id)


class VideoUploadAPIView(APIView):
    """Upload a video to the ``sadronevideo`` account and register a VideoEntity.

    Saving the VideoEntity fires the ``post_save`` signal, which runs the Azure
    ingestion/indexing pipeline (see ``apps/videos/signals.py``).

    Hardened against: oversized uploads, non-MP4/renamed/malformed files,
    disallowed codecs/resolution/duration, blob-path injection via the
    client filename, and information disclosure on error.
    """

    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "video-upload"

    def post(self, request, format=None):
        file_obj = request.FILES.get("file")
        raw_account_id = request.data.get("account_id")

        if not file_obj or not raw_account_id:
            return Response({"error": "file and account_id are required"},
                             status=status.HTTP_400_BAD_REQUEST)

        try:
            account_id = _validate_account_id(raw_account_id)
        except VideoValidationError:
            return Response({"error": "Invalid account_id."},
                             status=status.HTTP_400_BAD_REQUEST)

        # Declared size check (cheap, first line of defense — a lying
        # Content-Length is still bounded by the streamed-bytes check below).
        declared_size = getattr(file_obj, "size", None)
        if declared_size is not None and declared_size > MAX_UPLOAD_BYTES:
            return Response(
                {"error": f"File exceeds maximum size of {MAX_UPLOAD_BYTES} bytes."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        display_name = _sanitize_display_name(file_obj.name)

        with tempfile.TemporaryDirectory(prefix="upload_") as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "upload.mp4")

            try:
                written = self._write_capped(file_obj, tmp_path, MAX_UPLOAD_BYTES)
            except VideoValidationError as exc:
                return Response({"error": str(exc)},
                                 status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            finally:
                file_obj.seek(0)

            if written == 0:
                return Response({"error": "Uploaded file is empty."},
                                 status=status.HTTP_400_BAD_REQUEST)

            try:
                _sniff_mp4_ftyp(tmp_path)
                _probe_with_ffprobe(tmp_path)
            except VideoValidationError as exc:
                logger.info("rejecting upload %s: %s", display_name, exc)
                return Response({"error": f"Invalid MP4 file: {exc}"},
                                 status=status.HTTP_400_BAD_REQUEST)

            # Content-moderation screen (distinct from the structural/security
            # validation above). Intentionally fail-open per product policy:
            # a missing screening toolchain should not block legitimate
            # uploads. This is NOT a security control, so fail-open here is
            # acceptable in a way it would not be for the checks above.
            verdict = self._screen_video(tmp_path)
            if verdict in synth_screen.FABRICATED_VERDICTS:
                logger.info("rejecting upload %s: screened as %s", display_name, verdict)
                return Response(
                    {"error": "This video appears to be AI-generated or fabricated "
                              "and cannot be uploaded. Only camera-captured drone "
                              "footage is accepted.",
                     "verdict": verdict},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                sas_url = self._upload_to_blob(account_id, tmp_path, display_name)
            except Exception:  # noqa: BLE001
                logger.exception("video upload to blob storage failed")
                return Response({"error": "Upload failed. Please try again."},
                                 status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            video = VideoEntity()
            video.create_video(account_id=account_id, sas_url=sas_url)
            return Response(VideoEntitySerializer(video).data,
                             status=status.HTTP_201_CREATED)
        except Exception:  # noqa: BLE001
            logger.exception("video entity creation failed after successful upload")
            return Response({"error": "Upload succeeded but registration failed."},
                             status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _write_capped(file_obj, dest_path: str, max_bytes: int) -> int:
        """Stream ``file_obj`` to ``dest_path``, aborting if it exceeds
        ``max_bytes``. Guards against a spoofed/absent Content-Length by
        enforcing the cap against bytes actually written, not the
        declared size.
        """
        written = 0
        fd = os.open(dest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as out:
                for chunk in file_obj.chunks():
                    written += len(chunk)
                    if written > max_bytes:
                        raise VideoValidationError(
                            f"File exceeds maximum size of {max_bytes} bytes."
                        )
                    out.write(chunk)
        except VideoValidationError:
            raise
        return written

    @staticmethod
    def _screen_video(tmp_path: str) -> str:
        # Screening can be disabled entirely via settings; when off, uploads
        # bypass the synth screen and are treated as camera footage.
        if not getattr(settings, "VIDEO_SYNTH_SCREEN_ENABLED", True):
            return "camera"
        try:
            return synth_screen.screen(tmp_path).get("verdict", "camera")
        except Exception:  # noqa: BLE001
            logger.exception("video screening failed; allowing upload (fail-open)")
            return "camera"

    @staticmethod
    def _upload_to_blob(account_id: int, tmp_path: str, display_name: str) -> str:
        """Upload the validated temp file under a server-generated blob
        name (never the client-supplied filename) and return a
        short-lived read SAS URL.
        """
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas
        from core.azure.blob import service_client

        cfg = AzureEnvironmentConfig.from_settings()
        svc = service_client(cfg)

        safe_ext = ".mp4"
        blob_name = f"{account_id}/{uuid.uuid4().hex}{safe_ext}"

        with open(tmp_path, "rb") as fh:
            svc.get_blob_client(container=cfg.input_container, blob=blob_name).upload_blob(
                fh,
                overwrite=False,  # server-generated name is unique; never clobber
                metadata={"original_filename": display_name},
            )

        sas_token = generate_blob_sas(
            account_name=cfg.storage_account,
            container_name=cfg.input_container,
            blob_name=blob_name,
            account_key=cfg.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1),
        )
        return (f"https://{cfg.storage_account}.blob.core.windows.net/"
                f"{cfg.input_container}/{blob_name}?{sas_token}")


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


class FrameExtractAPIView(APIView):
    """Extract salient/stride JPEG frames from an account's video.

    A direct tool endpoint (peer of :class:`CornersAPIView`) that bypasses the
    query / RAG / agentic path. It writes frames into the *existing* ingestion
    blob layout (``{account_id}/images/{video_id}/frame{N}.jpg``, contiguous
    ``N`` from 0) and returns a **paginated** list of freshly minted read-only
    SAS URLs. Frame source, in order: already-uploaded frames (idempotent
    re-mint, no decode), else Azure Video Indexer key frames when configured,
    else a time stride (``stride`` seconds, default 10).

    Because frames land in the shared layout with contiguous numbering, the
    ingestion pipeline's ``get_uploaded_frames`` sees them and skips its own
    full-frame dump.

    Request (``IsAuthenticated``): ``account_id`` (required); optional
    ``video_id`` (a :class:`VideoEntity` pk) or ``video_sas_url``/``sas_url``;
    ``stride`` seconds; ``page``/``page_size`` (body or query params).
    """

    parser_classes = [MultiPartParser, FormParser, JSONParser]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, format=None):
        serializer = FrameExtractRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        account_id = serializer.validated_data["account_id"]
        video_id = serializer.validated_data.get("video_id")
        source_sas = serializer.resolved_sas_url()
        stride = float(serializer.validated_data.get("stride") or 10.0)

        entity = None
        if video_id:
            entity = VideoEntity.objects.filter(pk=video_id,
                                                account_id=account_id).first()
            if entity is None:
                return Response({"error": "video not found"},
                                status=status.HTTP_404_NOT_FOUND)
        elif not source_sas:
            entity = VideoEntity.objects.filter(
                account_id=account_id).order_by("-created").first()

        source_url = source_sas or (entity.sas_url if entity else None)
        if not source_url:
            return Response({"error": "no video found for account"},
                            status=status.HTTP_404_NOT_FOUND)

        video_pk = entity.pk if entity else 0
        # Blob-path segment: the VideoEntity pk, or omitted when only a raw SAS.
        path_segment = entity.pk if entity else None

        try:
            from apps.videos import frame_extract

            cfg = AzureEnvironmentConfig.from_settings()
            source, frames = frame_extract.extract_frames(
                cfg, source_url, account_id=str(account_id),
                video_id=path_segment, stride_seconds=stride)

            if entity is not None:
                self._sync_image_entities(entity, str(account_id), source_url, frames)

            paginator = StandardResultsSetPagination()
            self._inject_body_page_params(request)
            page = paginator.paginate_queryset(frames, request, view=self)
            items = FrameResultSerializer(page or [], many=True).data
            return Response({
                "account_id": str(account_id),
                "video_pk": video_pk,
                "source": source,
                "count": paginator.page.paginator.count,
                "next": paginator.get_next_link(),
                "previous": paginator.get_previous_link(),
                "results": items,
            }, status=status.HTTP_200_OK)
        except Exception as exc:  # noqa: BLE001
            logger.exception("frame extraction failed")
            return Response({"error": str(exc)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @staticmethod
    def _inject_body_page_params(request):
        """Let ``page``/``page_size`` arrive in the JSON body too, not only the
        query string (DRF pagination reads them from ``query_params``)."""
        q = request._request.GET.copy()
        for key in ("page", "page_size"):
            val = request.data.get(key)
            if key not in q and val not in (None, ""):
                q[key] = str(val)
        request._request.GET = q

    @staticmethod
    def _sync_image_entities(entity, account_id, source_url, frames):
        """Persist one :class:`ImageEntity` per frame (best-effort, idempotent)."""
        try:
            if entity.images.count() == len(frames):
                return
            entity.images.all().delete()
            rows = []
            for frame in frames:
                t = frame.get("t")
                ts = None
                if isinstance(t, (int, float)) and 0 <= t < 86400:
                    ts = (datetime.datetime.min + datetime.timedelta(seconds=float(t))).time()
                rows.append(ImageEntity(
                    video=entity, account_id=account_id,
                    index_name=(entity.index_name or ""), video_url=source_url,
                    sas_url=frame["sas_url"], timestamp=ts, status="extracted",
                ))
            ImageEntity.objects.bulk_create(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("frame ImageEntity persistence skipped: %s", exc)
