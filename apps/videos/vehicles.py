"""Vehicle tracking video for an account's uploaded drone video (synchronous).

Runs :func:`core.vehicles.pipeline.run_vehicle_pipeline` on the video behind a ``VideoEntity`` and
uploads the annotated MP4 and its JSON report next to the video's frames::

    {account_id}/images/{video_pk}/vehicles/vehicles_tracked.mp4
    {account_id}/images/{video_pk}/vehicles/vehicles_tracking.json

A time-range request (``start_s``/``end_s``) writes ``vehicles_tracked_<start>-<end>.mp4`` instead,
so a clip never masquerades as the full-video result. Existing results are reused unless ``force``.

:func:`track_vehicles` never raises: it returns ``{"state": "completed" | "failed" | "busy", ...}``.
``unavailable`` is true when the detector runtime (ultralytics/torch/weights) is missing; that is
checked before the video is downloaded. At most ``DVSA_VEHICLES_MAX_JOBS`` runs at once (GPU/RAM).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, Optional

from django.conf import settings

from core.azure.config import AzureEnvironmentConfig

logger = logging.getLogger("apps.videos")

COMPLETED, FAILED, BUSY, NONE = "completed", "failed", "busy", "none"

_guard = threading.Lock()
_slots: Optional[threading.BoundedSemaphore] = None


def _setting(name: str, default):
    value = getattr(settings, name, default)
    return default if value in (None, "") else value


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _range_suffix(start_s: Optional[float], end_s: Optional[float]) -> str:
    if start_s is None and end_s is None:
        return ""
    fmt = lambda v: f"{v:g}".replace(".", "p")  # noqa: E731
    return f"_{fmt(start_s or 0.0)}-{'end' if end_s is None else fmt(end_s)}"


def blob_names(account_id, video_pk, start_s=None, end_s=None) -> Dict[str, str]:
    base = f"{account_id}/images/{video_pk}/vehicles"
    sfx = _range_suffix(start_s, end_s)
    return {"video": f"{base}/vehicles_tracked{sfx}.mp4", "report": f"{base}/vehicles_tracking{sfx}.json"}


def pipeline_config(start_s: Optional[float] = None, end_s: Optional[float] = None):
    from core.vehicles.pipeline import VehiclePipelineConfig  # noqa: PLC0415
    from core.vehicles.tracker import TrackerConfig  # noqa: PLC0415

    return VehiclePipelineConfig(
        target_fps=float(_setting("DVSA_VEHICLES_TARGET_FPS", 15)),
        start_s=float(start_s or 0.0),
        end_s=end_s,
        max_seconds=float(_setting("DVSA_VEHICLES_MAX_SECONDS", 600)),
        motion_model=str(_setting("DVSA_VEHICLES_MOTION_MODEL", "similarity")),
        max_render_edge=int(_setting("DVSA_VEHICLES_MAX_RENDER_EDGE", 1920)),
        ffmpeg=_setting("DVSA_FFMPEG", None),
        tracker=TrackerConfig(low_thresh=float(_setting("DVSA_VEHICLES_CONF", 0.10))),
    )


def detector_config():
    from core.vehicles.detector import DetectorConfig  # noqa: PLC0415

    return DetectorConfig(
        weights=str(_setting("DVSA_VEHICLES_WEIGHTS", "yolo26x.pt")),
        model_dir=_setting("DVSA_VEHICLES_MODEL_DIR", None),
        device=_setting("DVSA_VEHICLES_DEVICE", None),
        imgsz=int(_setting("DVSA_VEHICLES_IMGSZ", 0)),
        conf=float(_setting("DVSA_VEHICLES_CONF", 0.10)),
        tile=_truthy(_setting("DVSA_VEHICLES_TILE", "false")),
    )


def make_detector(frame):
    from core.vehicles.detector import UltralyticsVehicleDetector  # noqa: PLC0415

    return UltralyticsVehicleDetector(detector_config(), warmup_shape=frame.shape)


def _sas_hours() -> int:
    return int(_setting("DVSA_VEHICLES_SAS_HOURS", 24))


def _blob_exists(config: AzureEnvironmentConfig, name: str) -> bool:
    from core.azure import blob as azure_blob  # noqa: PLC0415

    try:
        return azure_blob.service_client(config).get_blob_client(
            container=config.input_container, blob=name).exists()
    except Exception as exc:  # noqa: BLE001 - no storage credentials etc. => treat as absent
        logger.info("vehicles blob check failed: %s", exc)
        return False


def existing_result(account_id, video_pk, start_s=None, end_s=None,
                    config: Optional[AzureEnvironmentConfig] = None) -> Optional[Dict[str, Any]]:
    """Fresh read-SAS URLs for an already generated result, or ``None``."""
    from core.azure import blob as azure_blob  # noqa: PLC0415

    try:
        config = config or AzureEnvironmentConfig.from_settings()
    except Exception as exc:  # noqa: BLE001
        logger.info("vehicles config unavailable: %s", exc)
        return None
    names = blob_names(account_id, video_pk, start_s, end_s)
    if not _blob_exists(config, names["video"]):
        return None
    out = {"state": COMPLETED, "blob": names["video"],
           "downloadUrl": azure_blob.read_sas_for_blob(config, names["video"], ttl_hours=_sas_hours())}
    if _blob_exists(config, names["report"]):
        out["reportUrl"] = azure_blob.read_sas_for_blob(config, names["report"], ttl_hours=_sas_hours())
    return out


def fresh_source_url(config: AzureEnvironmentConfig, source_url: str) -> str:
    """Re-sign a blob URL in our own storage account with a fresh read SAS.

    ``VideoEntity.sas_url`` carries the SAS minted at upload time, which expires (weeks later every
    download would fail with 403). When the URL points at ``config.storage_account`` and the
    account key is configured, mint a new short-lived read SAS for the same blob; otherwise return
    the URL unchanged.
    """
    from urllib.parse import unquote, urlparse  # noqa: PLC0415

    from core.azure import blob as azure_blob  # noqa: PLC0415

    try:
        parsed = urlparse(source_url)
        account = getattr(config, "storage_account", None)
        own_host = f"{account}.blob.core.windows.net"
        if not account or not getattr(config, "account_key", None) or parsed.hostname != own_host:
            return source_url
        container, _, blob_name = unquote(parsed.path).lstrip("/").partition("/")
        if not container or not blob_name:
            return source_url
        return azure_blob.read_sas_for_blob(config, blob_name, container=container, ttl_hours=2)
    except Exception as exc:  # noqa: BLE001 - fall back to the stored URL
        logger.info("could not re-sign source url: %s", exc)
        return source_url


def _semaphore() -> threading.BoundedSemaphore:
    global _slots
    with _guard:
        if _slots is None:
            _slots = threading.BoundedSemaphore(max(1, int(_setting("DVSA_VEHICLES_MAX_JOBS", 1))))
        return _slots


def _summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return {k: report.get(k) for k in ("video", "vehicles", "output", "timing_s", "performance")}


def track_vehicles(account_id, video_pk, source_url: str, *, force: bool = False,
                   start_s: Optional[float] = None, end_s: Optional[float] = None,
                   config: Optional[AzureEnvironmentConfig] = None) -> Dict[str, Any]:
    """Generate (or reuse) the vehicle tracking video for one ``VideoEntity``. Never raises."""
    from core.azure import blob as azure_blob  # noqa: PLC0415
    from core.vehicles.detector import VehiclesUnavailable, runtime_missing  # noqa: PLC0415
    from core.vehicles.pipeline import REPORT_NAME, VIDEO_NAME, run_vehicle_pipeline  # noqa: PLC0415

    if not force:
        cached = existing_result(account_id, video_pk, start_s, end_s, config)
        if cached:
            return {**cached, "cached": True}

    slots = _semaphore()
    if not slots.acquire(blocking=False):
        return {"state": BUSY, "reason": "another vehicle tracking job is running; retry later"}
    video_path = work_dir = None
    try:
        missing = runtime_missing()
        if missing:  # fail fast, before downloading a large video
            raise VehiclesUnavailable(f"{missing} is not installed (pip install -r requirements/vehicles.txt)")
        config = config or AzureEnvironmentConfig.from_settings()
        t0 = time.perf_counter()
        video_path = azure_blob.download_blob_to_temp(fresh_source_url(config, source_url))
        download_s = time.perf_counter() - t0
        work_dir = tempfile.mkdtemp(prefix="dvsa-vehicles-")
        report = run_vehicle_pipeline(video_path, work_dir, pipeline_config(start_s, end_s), make_detector)

        names = blob_names(account_id, video_pk, start_s, end_s)
        t0 = time.perf_counter()
        with open(os.path.join(work_dir, VIDEO_NAME), "rb") as fh:
            download_url = azure_blob.put_blob_and_sas(config, names["video"], fh, content_type="video/mp4",
                                                       ttl_hours=_sas_hours())
        with open(os.path.join(work_dir, REPORT_NAME), "rb") as fh:
            report_url = azure_blob.put_blob_and_sas(config, names["report"], fh, content_type="application/json",
                                                     ttl_hours=_sas_hours())
        summary = _summary(report)
        if isinstance(summary.get("timing_s"), dict):
            summary["timing_s"] = {**summary["timing_s"], "download": round(download_s, 3),
                                   "upload": round(time.perf_counter() - t0, 3)}
        logger.info("vehicle tracking for video %s uploaded to %s", video_pk, names["video"])
        return {"state": COMPLETED, "cached": False, "blob": names["video"], "downloadUrl": download_url,
                "reportUrl": report_url, "summary": summary}
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never raised
        unavailable = isinstance(exc, VehiclesUnavailable)
        logger.warning("vehicle tracking for video %s %s: %s", video_pk,
                       "unavailable" if unavailable else "failed", exc, exc_info=not unavailable)
        return {"state": FAILED, "unavailable": unavailable, "reason": str(exc)[:500]}
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as exc:
                logger.info("vehicles temp cleanup failed: %s", exc)
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        slots.release()
