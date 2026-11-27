"""Direct-to-cloud upload API (two-step presigned upload + completion + storage events).

Ownership is the authenticated Django ``User``: the UI's ``account_id`` is ``str(user.pk)``,
so an ``account_id`` that names anyone else is rejected with 403.
"""

import hmac
import logging

from django.conf import settings
from django.shortcuts import get_object_or_404
from drf_spectacular.utils import extend_schema
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.jobs.serializers import AnalyticsJobSerializer
from apps.jobs.services import UnsupportedJobType
from core.storage import ObjectInfo, StorageEventError, get_object_store

from . import services
from .models import MediaAsset
from .serializers import (CompleteRequestSerializer, MediaAssetSerializer,
                          UploadDestinationRequestSerializer)

logger = logging.getLogger("apps.media")

IDEMPOTENCY_HEADER = "Idempotency-Key"


def _account_mismatch(request, account_id):
    if account_id not in (None, "") and str(account_id) != str(request.user.pk):
        return Response({"error": "account_id does not belong to the authenticated user"},
                        status=status.HTTP_403_FORBIDDEN)
    return None


def _error(exc: services.UploadError):
    return Response({"error": str(exc), **exc.extra}, status=exc.status)


class _OwnedMedia:
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return MediaAsset.objects.filter(owner=self.request.user)


class UploadDestinationView(APIView):
    """Step 1: create a PENDING asset and return a write-only upload URL for its one key.

    Send an ``Idempotency-Key`` header to make client retries return the same asset.
    """

    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(request=UploadDestinationRequestSerializer, responses={201: dict, 200: dict})
    def post(self, request):
        serializer = UploadDestinationRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data
        denied = _account_mismatch(request, data.get("account_id"))
        if denied:
            return denied
        key = (request.headers.get(IDEMPOTENCY_HEADER) or "").strip()[:128]
        try:
            asset, created = services.create_upload(request.user, data, key)
            body = services.destination(asset)
        except services.UploadError as exc:
            return _error(exc)
        return Response(body, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class UploadDestinationRenewView(_OwnedMedia, APIView):
    """Re-issue the upload URL for a PENDING asset (e.g. after the first one expired)."""

    @extend_schema(request=None, responses={200: dict})
    def post(self, request, pk):
        asset = get_object_or_404(self.get_queryset(), pk=pk)
        if asset.upload_status != MediaAsset.UploadStatus.PENDING:
            return Response({"error": f"media is already {asset.upload_status}"},
                            status=status.HTTP_409_CONFLICT)
        return Response(services.destination(asset), status=status.HTTP_200_OK)


class MediaListView(_OwnedMedia, generics.ListAPIView):
    serializer_class = MediaAssetSerializer
    filterset_fields = ("upload_status", "flight_id")


class MediaDetailView(_OwnedMedia, generics.RetrieveAPIView):
    serializer_class = MediaAssetSerializer


class MediaCompleteView(_OwnedMedia, APIView):
    """Step 2: verify the object exists, mark it UPLOADED and queue its analytics job.

    Videos return ``202`` with ``job_id`` (``job_type`` defaults to ``EXTRACT_FRAMES``);
    sidecars (``.srt``/``.json``) return ``200``. Repeating the call returns the same job.
    """

    @extend_schema(request=CompleteRequestSerializer, responses={202: dict, 200: dict})
    def post(self, request, pk):
        asset = get_object_or_404(self.get_queryset(), pk=pk)
        serializer = CompleteRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data
        denied = _account_mismatch(request, data.get("account_id"))
        if denied:
            return denied
        try:
            asset, job, _ = services.complete_upload(asset, job_type=data.get("job_type"),
                                                     job_parameters=data.get("parameters"))
        except UnsupportedJobType as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except services.UploadError as exc:
            return _error(exc)
        body = {"media_id": str(asset.id), "upload_status": asset.upload_status,
                "file_size_bytes": asset.file_size_bytes, "etag": asset.etag}
        if job is None:
            return Response(body, status=status.HTTP_200_OK)
        body.update({"job_id": str(job.id), "job": AnalyticsJobSerializer(job).data})
        return Response(body, status=status.HTTP_202_ACCEPTED)


class StorageEventView(APIView):
    """Provider "object created" push (Azure Event Grid ``BlobCreated`` today).

    Authenticated by ``?token=`` matching ``DVSA_STORAGE_EVENT_TOKEN`` (disabled when that
    is empty). Completes matching PENDING assets exactly as the client callback would.
    """

    authentication_classes = []
    permission_classes = [permissions.AllowAny]
    throttle_classes = []

    @extend_schema(request=dict, responses={200: dict})
    def post(self, request, provider):
        expected = str(getattr(settings, "DVSA_STORAGE_EVENT_TOKEN", "") or "")
        supplied = str(request.query_params.get("token", ""))
        if not expected or not hmac.compare_digest(expected, supplied):
            return Response({"error": "forbidden"}, status=status.HTTP_403_FORBIDDEN)
        try:
            store = get_object_store(provider)
        except ValueError:
            return Response({"error": "unknown provider"}, status=status.HTTP_404_NOT_FOUND)
        try:
            created, handshake = store.parse_created_events(request.data, dict(request.headers))
        except StorageEventError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if handshake is not None:
            return Response(handshake, status=status.HTTP_200_OK)
        completed = []
        for event in created:
            asset = MediaAsset.objects.filter(object_key=event.key, storage_provider=store.provider,
                                              bucket=store.bucket).first()
            if asset is None or asset.upload_status != MediaAsset.UploadStatus.PENDING:
                continue
            info = ObjectInfo(key=event.key, size=event.size, etag=event.etag or "") \
                if event.size is not None else None
            try:
                asset, job, won = services.complete_upload(asset, info=info)
            except (services.UploadError, UnsupportedJobType) as exc:
                logger.info("storage event for %s not completed: %s", event.key, exc)
                continue
            if won:
                completed.append({"media_id": str(asset.id), "job_id": str(job.id) if job else None})
        return Response({"received": len(created), "completed": completed}, status=status.HTTP_200_OK)
