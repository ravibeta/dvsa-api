"""Analytics job API: create, poll, list (incremental), cancel and retry (owner-scoped)."""

import datetime
import logging

import django_filters
from django.shortcuts import get_object_or_404
from drf_spectacular.utils import extend_schema
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.media.models import MediaAsset

from .models import AnalyticsJob
from .serializers import AnalyticsJobSerializer, JobCreateSerializer
from .services import cancel_job, create_job, retry_job

logger = logging.getLogger("apps.jobs")

OUTPUT_URL_TTL = datetime.timedelta(hours=1)
MAX_FRAME_URLS = 200


class JobFilter(django_filters.FilterSet):
    status = django_filters.CharFilter(field_name="status", lookup_expr="iexact")
    job_type = django_filters.CharFilter(field_name="job_type", lookup_expr="iexact")
    media_id = django_filters.UUIDFilter(field_name="media_asset_id")
    updated_after = django_filters.IsoDateTimeFilter(field_name="updated_at", lookup_expr="gt")

    class Meta:
        model = AnalyticsJob
        fields = ("status", "job_type", "media_id", "updated_after")


class _OwnedJobs:
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return AnalyticsJob.objects.filter(owner=self.request.user).select_related("media_asset")


def output_urls(summary):
    """Fresh read URLs for a job's output keys (keys, not URLs, are what is stored)."""
    outputs = (summary or {}).get("outputs") or {}
    if not outputs:
        return {}
    from core.storage import get_object_store  # noqa: PLC0415

    try:
        store = get_object_store()
        urls = {}
        for name, out in outputs.items():
            if out.get("key"):
                urls[name] = store.presign_read(out["key"], OUTPUT_URL_TTL)
            elif out.get("keys"):
                urls[name] = [store.presign_read(k, OUTPUT_URL_TTL) for k in out["keys"][:MAX_FRAME_URLS]]
        return urls
    except Exception as exc:  # noqa: BLE001 - URLs are a convenience; the keys are in the summary
        logger.info("could not sign job output URLs: %s", exc)
        return {}


class JobListCreateView(_OwnedJobs, generics.ListAPIView):
    """``GET`` paginated jobs (filters: ``status``, ``job_type``, ``media_id``, ``updated_after``,
    ordered by ``updated_at`` then ``id`` for incremental sync); ``POST`` starts a job."""

    serializer_class = AnalyticsJobSerializer
    filterset_class = JobFilter
    ordering_fields = ("updated_at", "created_at")

    def get_queryset(self):
        return super().get_queryset().order_by("updated_at", "id")

    @extend_schema(request=JobCreateSerializer, responses={202: AnalyticsJobSerializer})
    def post(self, request):
        serializer = JobCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data
        asset = get_object_or_404(MediaAsset, pk=data["media_id"], owner=request.user)
        if asset.upload_status != MediaAsset.UploadStatus.UPLOADED or not asset.is_video:
            return Response({"error": "media must be an uploaded video"},
                            status=status.HTTP_409_CONFLICT)
        job = create_job(request.user, asset, data["job_type"],
                         model_name=data.get("model_name", ""),
                         model_version=data.get("model_version", ""),
                         parameters=data.get("parameters") or {})
        return Response(AnalyticsJobSerializer(job).data, status=status.HTTP_202_ACCEPTED)


class JobDetailView(_OwnedJobs, generics.RetrieveAPIView):
    serializer_class = AnalyticsJobSerializer

    def retrieve(self, request, *args, **kwargs):
        job = self.get_object()
        data = AnalyticsJobSerializer(job).data
        if job.status == AnalyticsJob.Status.COMPLETED:
            data["download_urls"] = output_urls(job.result_summary)
        return Response(data)


class JobCancelView(_OwnedJobs, APIView):
    @extend_schema(request=None, responses={200: AnalyticsJobSerializer})
    def post(self, request, pk):
        job = get_object_or_404(self.get_queryset(), pk=pk)
        if not cancel_job(job):
            job.refresh_from_db()
            return Response({"error": f"job is already {job.status}"}, status=status.HTTP_409_CONFLICT)
        job.refresh_from_db()
        return Response(AnalyticsJobSerializer(job).data, status=status.HTTP_200_OK)


class JobRetryView(_OwnedJobs, APIView):
    @extend_schema(request=None, responses={202: AnalyticsJobSerializer})
    def post(self, request, pk):
        job = get_object_or_404(self.get_queryset(), pk=pk)
        retried = retry_job(job)
        if retried is None:
            job.refresh_from_db()
            return Response({"error": f"only FAILED or CANCELLED jobs can be retried (job is {job.status})"},
                            status=status.HTTP_409_CONFLICT)
        return Response(AnalyticsJobSerializer(retried).data, status=status.HTTP_202_ACCEPTED)
