"""Two-step direct upload: issue a write-only upload URL, then confirm and dispatch a job.

Cloud-agnostic: everything goes through :func:`core.storage.get_object_store`.
"""
from __future__ import annotations

import datetime
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from django.conf import settings
from django.utils import timezone

from core.storage import ObjectInfo, get_object_store

from . import telemetry
from .models import MediaAsset

logger = logging.getLogger("apps.media")

ALLOWED_TYPES = {
    ".mp4": {"video/mp4"},
    ".mov": {"video/quicktime"},
    ".srt": {"application/x-subrip", "text/srt", "text/plain"},
    ".json": {"application/json"},
}
SIDECAR_MAX_BYTES = 50 * 1024 * 1024
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(Exception):
    """A client-facing upload error with an HTTP status."""

    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra


def upload_ttl() -> datetime.timedelta:
    return datetime.timedelta(minutes=int(getattr(settings, "DVSA_UPLOAD_SAS_MINUTES", 30)))


def max_upload_bytes() -> int:
    return int(getattr(settings, "DVSA_UPLOAD_MAX_BYTES", 20 * 1024 ** 3))


def safe_filename(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    stem, ext = os.path.splitext(base)
    stem = _UNSAFE.sub("_", stem).strip("._")[:120] or "upload"
    return f"{stem}{ext.lower()}"


def validate_file(filename: str, content_type: str, size: int) -> str:
    """Return the sanitised filename or raise :class:`UploadError`."""
    clean = safe_filename(filename)
    ext = os.path.splitext(clean)[1]
    allowed = ALLOWED_TYPES.get(ext)
    if allowed is None:
        raise UploadError(f"unsupported file type {ext or '(none)'}; allowed: {', '.join(ALLOWED_TYPES)}")
    if (content_type or "").split(";")[0].strip().lower() not in allowed:
        raise UploadError(f"content_type for {ext} must be one of: {', '.join(sorted(allowed))}")
    if size <= 0 or size > max_upload_bytes():
        raise UploadError(f"file_size_bytes must be between 1 and {max_upload_bytes()}")
    if ext not in (".mp4", ".mov") and size > SIDECAR_MAX_BYTES:
        raise UploadError(f"{ext} sidecars are limited to {SIDECAR_MAX_BYTES} bytes")
    return clean


def object_key(owner_id, flight_id: str, media_id, filename: str) -> str:
    return f"{owner_id}/uploads/{flight_id or '_'}/{media_id}/{filename}"


def destination(asset: MediaAsset, store=None) -> Dict[str, Any]:
    store = store or get_object_store()
    target = store.presign_upload(asset.object_key, asset.content_type, upload_ttl())
    return {"media_id": str(asset.id), "upload_url": target.url, "method": target.method,
            "headers": target.headers, "expires_at": target.expires_at.isoformat(),
            "object_key": asset.object_key, "blob_path": asset.object_key,
            "upload_status": asset.upload_status}


def create_upload(owner, data: Dict[str, Any], idempotency_key: str = "") -> Tuple[MediaAsset, bool]:
    """``(asset, created)``. Repeating a request with the same ``Idempotency-Key`` returns
    the same PENDING asset instead of creating a duplicate."""
    filename = validate_file(data["filename"], data["content_type"], int(data["file_size_bytes"]))
    if idempotency_key:
        existing = MediaAsset.objects.filter(owner=owner, idempotency_key=idempotency_key).first()
        if existing is not None:
            if existing.filename != filename or existing.file_size_bytes != int(data["file_size_bytes"]):
                raise UploadError("Idempotency-Key was already used for a different file", 409)
            if existing.upload_status != MediaAsset.UploadStatus.PENDING:
                raise UploadError(f"media is already {existing.upload_status}", 409,
                                  media_id=str(existing.id))
            return existing, False
    store = get_object_store()
    asset = MediaAsset(owner=owner, flight_id=data.get("flight_id") or "", filename=filename,
                       content_type=data["content_type"].split(";")[0].strip().lower(),
                       file_size_bytes=int(data["file_size_bytes"]),
                       storage_provider=store.provider, bucket=store.bucket,
                       checksum_sha256=(data.get("checksum_sha256") or "").lower(),
                       metadata=data.get("metadata") or {}, idempotency_key=idempotency_key)
    asset.object_key = object_key(owner.pk, asset.flight_id, asset.id, filename)
    asset.save()
    return asset, True


def _capture_telemetry(asset: MediaAsset, store) -> None:
    if asset.extension != ".srt":
        return
    try:
        text = store.read_bytes(asset.object_key, SIDECAR_MAX_BYTES).decode("utf-8", errors="ignore")
        samples = telemetry.parse_srt(text)
    except Exception as exc:  # noqa: BLE001 - telemetry is optional context
        logger.info("telemetry capture for %s skipped: %s", asset.id, exc)
        return
    if samples:
        asset.metadata = {**(asset.metadata or {}), "telemetry": samples}
        MediaAsset.objects.filter(pk=asset.pk).update(metadata=asset.metadata)


def latest_job(asset: MediaAsset):
    return asset.jobs.order_by("created_at").first()


def complete_upload(asset: MediaAsset, *, info: Optional[ObjectInfo] = None,
                    job_type: Optional[str] = None, job_parameters: Optional[Dict[str, Any]] = None
                    ) -> Tuple[MediaAsset, Any, bool]:
    """Confirm an upload exists, mark it UPLOADED and (for videos) queue its job.

    Returns ``(asset, job_or_None, newly_completed)``. Idempotent: a repeated call (or a
    storage event racing the client's call) returns the job the first completion created.
    """
    from apps.jobs.services import create_job, validate_job_type  # noqa: PLC0415
    from apps.notifications.broadcast import broadcast_live_alert  # noqa: PLC0415
    from apps.notifications.events import MEDIA_UPLOADED  # noqa: PLC0415

    if asset.is_video:
        job_type = validate_job_type(job_type)
    if asset.upload_status == MediaAsset.UploadStatus.UPLOADED:
        return asset, latest_job(asset), False
    if asset.upload_status == MediaAsset.UploadStatus.FAILED:
        raise UploadError(f"upload failed: {asset.error_message}", 409)

    store = get_object_store(asset.storage_provider)
    if info is None or info.size is None:
        info = store.head(asset.object_key)
    if info is None:
        raise UploadError("object not found in storage; upload it before completing", 409)
    if int(info.size) != asset.file_size_bytes:
        reason = f"uploaded size {info.size} does not match declared file_size_bytes {asset.file_size_bytes}"
        MediaAsset.objects.filter(pk=asset.pk, upload_status=MediaAsset.UploadStatus.PENDING).update(
            upload_status=MediaAsset.UploadStatus.FAILED, error_message=reason, updated_at=timezone.now())
        raise UploadError(reason, 422)

    now = timezone.now()
    won = MediaAsset.objects.filter(pk=asset.pk, upload_status=MediaAsset.UploadStatus.PENDING).update(
        upload_status=MediaAsset.UploadStatus.UPLOADED, etag=info.etag or "", uploaded_at=now,
        updated_at=now) == 1
    asset.refresh_from_db()
    if not won:  # someone else completed it first
        return asset, latest_job(asset), False

    _capture_telemetry(asset, store)
    broadcast_live_alert(asset.owner_id, MEDIA_UPLOADED, {
        "media_id": str(asset.id), "flight_id": asset.flight_id, "filename": asset.filename,
        "file_size_bytes": asset.file_size_bytes, "object_key": asset.object_key})
    job = None
    if asset.is_video:
        job = create_job(asset.owner, asset, job_type, parameters=job_parameters or {})
    return asset, job, True
