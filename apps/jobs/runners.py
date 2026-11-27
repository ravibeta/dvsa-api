"""Job-type runners: adapt the existing per-video services to uploaded media assets.

Each runner reuses an existing service unchanged and keys its outputs by the
``MediaAsset`` id (never a ``VideoEntity`` pk). No ``VideoEntity`` is created, because its
``post_save`` starts the costly indexer workflow, which is reserved for per-video
``FULL_PIPELINE`` requests.

* ``VEHICLE_TRACKING``    -> :func:`apps.videos.vehicles.track_vehicles`
* ``GEOSPATIAL_MAPPING``  -> :func:`apps.videos.orthomosaic_jobs.run_job`
* ``EXTRACT_FRAMES``      -> :func:`apps.videos.frame_extract.extract_frames` (``/extract-frames/``)

The services are Azure-only today, so this module is the one place that would route to
another cloud's equivalents. Asset bytes are reached through the cloud-agnostic store.
"""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from django.conf import settings

from apps.media import telemetry
from apps.media.models import MediaAsset

from .cancel import JobCancelToken
from .models import AnalyticsJob

logger = logging.getLogger("apps.jobs")

JT = AnalyticsJob.JobType
READ_TTL = datetime.timedelta(hours=2)
REPORT_MAX_BYTES = 64 * 1024 * 1024


class SlotBusy(Exception):
    """The pipeline's concurrency slot is taken; the job is re-queued with a delay."""


class JobFailed(Exception):
    pass


@dataclass
class JobContext:
    job: AnalyticsJob
    asset: MediaAsset
    store: Any
    cancel: JobCancelToken
    progress: Callable[[int, str], None]

    @property
    def account_id(self) -> str:
        return str(self.job.owner_id)

    @property
    def video_key(self) -> str:
        return str(self.asset.id)

    @property
    def params(self) -> Dict[str, Any]:
        return self.job.parameters or {}

    def read_url(self) -> str:
        return self.store.presign_read(self.asset.object_key, READ_TTL)


def _float(value) -> Optional[float]:
    return None if value in (None, "") else float(value)


# --------------------------------------------------------------------- telemetry / GeoJSON
def asset_telemetry(asset: MediaAsset) -> List[Dict[str, float]]:
    """Drone positions captured for this video: its own metadata, else an uploaded ``.srt``
    sidecar with the same owner and ``flight_id``."""
    samples = telemetry.from_metadata(asset.metadata)
    if samples or not asset.flight_id:
        return samples
    sidecar = (MediaAsset.objects.filter(owner_id=asset.owner_id, flight_id=asset.flight_id,
                                         upload_status=MediaAsset.UploadStatus.UPLOADED,
                                         filename__iendswith=".srt")
               .order_by("-uploaded_at").first())
    return list((sidecar.metadata or {}).get("telemetry") or []) if sidecar else []


def detections_geojson(tracks: List[Dict[str, Any]], samples: List[Dict[str, float]]
                       ) -> Dict[str, Any]:
    """One Point per confirmed track: the drone position nearest the track's first sighting
    when telemetry was captured, else the reference-plane pixel position (``crs: pixel``)."""
    features = []
    for tr in tracks:
        trajectory = tr.get("trajectory_ref_px") or []
        first_t = tr.get("first_t", trajectory[0][0] if trajectory else 0.0)
        props = {"track_id": tr.get("id"), "class": tr.get("class"),
                 "state": tr.get("dominant_state"),
                 "confidence": (tr.get("uncertainty") or {}).get("mean_conf"),
                 "first_t": first_t, "last_t": tr.get("last_t")}
        coords = None
        if samples:
            fix = telemetry.nearest(samples, float(first_t or 0.0))
            coords = [fix["lon"], fix["lat"]] + ([fix["alt"]] if "alt" in fix else [])
            props["position_source"] = "drone_telemetry"
        elif trajectory:
            coords = [trajectory[0][1], trajectory[0][2]]
            props["position_source"] = "pixel"
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Point", "coordinates": coords} if coords else None})
    collection: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if not samples:
        collection["crs"] = {"type": "name", "properties": {"name": "pixel"}}
    return collection


# --------------------------------------------------------------------------------- runners
def run_vehicle_tracking(ctx: JobContext) -> Dict[str, Any]:
    from apps.videos import vehicles  # noqa: PLC0415

    start_s, end_s = _float(ctx.params.get("start_s")), _float(ctx.params.get("end_s"))
    ctx.progress(10, "tracking vehicles")
    result = vehicles.track_vehicles(ctx.account_id, ctx.video_key, ctx.read_url(),
                                     force=bool(ctx.params.get("force")), start_s=start_s, end_s=end_s)
    if result.get("state") == vehicles.BUSY:
        raise SlotBusy(result.get("reason", "busy"))
    if result.get("state") != vehicles.COMPLETED:
        raise JobFailed(result.get("reason") or "vehicle tracking failed")
    ctx.cancel.raise_if_set()
    ctx.progress(85, "building detections")
    names = vehicles.blob_names(ctx.account_id, ctx.video_key, start_s, end_s)
    report = json.loads(ctx.store.read_bytes(names["report"], REPORT_MAX_BYTES))
    tracks = report.get("tracks") or []
    return {
        "outputs": {"video": {"key": names["video"], "content_type": "video/mp4"},
                    "report": {"key": names["report"], "content_type": "application/json"}},
        "summary": result.get("summary") or {},
        "cached": bool(result.get("cached")),
        "geojson": detections_geojson(tracks, asset_telemetry(ctx.asset)),
    }


def run_geospatial_mapping(ctx: JobContext) -> Dict[str, Any]:
    from apps.videos import orthomosaic_jobs  # noqa: PLC0415

    ctx.progress(10, "building orthomosaic")
    state = orthomosaic_jobs.run_job(ctx.account_id, ctx.video_key, ctx.read_url())
    if state.get("state") != orthomosaic_jobs.COMPLETED:
        raise JobFailed(state.get("reason") or "orthomosaic failed")
    return {"outputs": {"orthomosaic": {"key": state.get("blob"), "content_type": "image/jpeg"}},
            "summary": state.get("output") or {}}


def run_extract_frames(ctx: JobContext) -> Dict[str, Any]:
    from apps.videos import frame_extract  # noqa: PLC0415
    from core.azure.config import AzureEnvironmentConfig  # noqa: PLC0415

    stride = float(ctx.params.get("stride") or 10.0)
    ctx.progress(10, "extracting frames")
    # video_id=None: frames land beside the upload ({...}/{media_id}/images/frameN.jpg).
    source, frames = frame_extract.extract_frames(
        AzureEnvironmentConfig.from_settings(), ctx.read_url(), account_id=ctx.account_id,
        video_id=None, stride_seconds=stride)
    return {"outputs": {"frames": {"keys": [f["blob_name"] for f in frames],
                                   "content_type": "image/jpeg"}},
            "summary": {"source": source, "count": len(frames), "stride_s": stride,
                        "timestamps": [f.get("t") for f in frames]}}


RUNNERS: Dict[str, Callable[[JobContext], Dict[str, Any]]] = {
    JT.VEHICLE_TRACKING: run_vehicle_tracking,
    JT.GEOSPATIAL_MAPPING: run_geospatial_mapping,
    JT.EXTRACT_FRAMES: run_extract_frames,
}


def run(ctx: JobContext) -> Dict[str, Any]:
    runner = RUNNERS.get(ctx.job.job_type)
    if runner is None:
        raise JobFailed(f"job type {ctx.job.job_type} is not supported for bulk jobs")
    ctx.cancel.raise_if_set()
    summary = runner(ctx)
    ctx.cancel.raise_if_set()
    return summary


def min_vehicles_for_alert() -> int:
    return int(getattr(settings, "DVSA_ALERT_MIN_VEHICLES", 1))
