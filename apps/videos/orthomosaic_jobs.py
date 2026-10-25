"""Background orthomosaic jobs for uploaded videos.

An orthomosaic takes minutes of CPU/RAM in an ODM container, so it never runs inside a
request unless the caller explicitly asks to wait. :func:`start_job` runs it on a daemon
thread and records progress; the result is the blob
``{account_id}/images/{video_pk}/orthomosaic.jpg`` in the shared frame layout.

Isolation guarantees (the job must never affect the rest of dvsa-api):

* every failure -- no Docker, no ODM image, ODM error, storage error -- is caught, logged and
  recorded as ``failed``; nothing is raised into the caller (an upload, a signal handler);
* the engine is checked *before* the video is downloaded, so a box without Docker fails fast;
* at most ``DVSA_ORTHOMOSAIC_MAX_JOBS`` (default 1) run at once, one job per video at a time;
* the blob itself is the source of truth for "completed", so state survives restarts.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from typing import Any, Dict, Optional

from django.conf import settings
from django.core.cache import cache

from core.azure.config import AzureEnvironmentConfig

logger = logging.getLogger("apps.videos")

QUEUED, RUNNING, COMPLETED, FAILED = "queued", "running", "completed", "failed"
_STATE_TTL = 24 * 3600

_guard = threading.Lock()
_active: set = set()
_slots: Optional[threading.BoundedSemaphore] = None


def auto_enabled() -> bool:
    return str(getattr(settings, "DVSA_ORTHOMOSAIC_AUTO", "true")).strip().lower() not in (
        "", "0", "false", "no", "off", "none")


def _state_key(account_id, video_pk) -> str:
    return f"orthomosaic:{account_id}:{video_pk}"


def get_state(account_id, video_pk) -> Dict[str, Any]:
    return cache.get(_state_key(account_id, video_pk)) or {"state": "none"}


def _set_state(account_id, video_pk, state: str, **extra) -> Dict[str, Any]:
    value = {"state": state, **extra}
    cache.set(_state_key(account_id, video_pk), value, _STATE_TTL)
    return value


def build_options() -> Dict[str, Any]:
    """Pipeline options from settings (``DVSA_ORTHOMOSAIC_*``); defaults match the script."""
    g = lambda name, default: getattr(settings, name, default)  # noqa: E731
    return dict(
        long_edge=int(g("DVSA_ORTHOMOSAIC_LONG_EDGE", 6000)),
        max_megapixels=float(g("DVSA_ORTHOMOSAIC_MAX_MEGAPIXELS", 64)),
        interval=float(g("DVSA_ORTHOMOSAIC_INTERVAL", 1)),
        max_frame_size=int(g("DVSA_ORTHOMOSAIC_MAX_FRAME_SIZE", 2560)),
        backend=str(g("DVSA_ORTHOMOSAIC_BACKEND", "docker")),
        odm_image=str(g("DVSA_ORTHOMOSAIC_IMAGE", "opendronemap/odm:latest")),
        odm_run=g("DVSA_ORTHOMOSAIC_ODM_RUN", None) or None,
        concurrency=int(g("DVSA_ORTHOMOSAIC_CONCURRENCY", 4)),
        fast=str(g("DVSA_ORTHOMOSAIC_FAST", "false")).strip().lower() in ("1", "true", "yes", "on"),
    )


def blob_exists(config: AzureEnvironmentConfig, blob_name: str) -> bool:
    from core.azure import blob as azure_blob  # noqa: PLC0415

    try:
        return azure_blob.service_client(config).get_blob_client(
            container=config.input_container, blob=blob_name).exists()
    except Exception as exc:  # noqa: BLE001 - no storage creds etc. => not available
        logger.info("orthomosaic blob check failed: %s", exc)
        return False


def result_url(config: AzureEnvironmentConfig, account_id, video_pk) -> Optional[str]:
    """Fresh read-SAS URL for an existing orthomosaic, or ``None``."""
    from core.azure import blob as azure_blob  # noqa: PLC0415
    from core.azure.orthomosaic import orthomosaic_blob_name  # noqa: PLC0415

    name = orthomosaic_blob_name(account_id, video_pk)
    if not blob_exists(config, name):
        return None
    return azure_blob.read_sas_for_blob(config, name)


def status(account_id, video_pk, config: Optional[AzureEnvironmentConfig] = None) -> Dict[str, Any]:
    """Job state, resolving ``completed`` from the blob (survives restarts / other workers)."""
    state = dict(get_state(account_id, video_pk))
    try:
        config = config or AzureEnvironmentConfig.from_settings()
        url = result_url(config, account_id, video_pk)
    except Exception as exc:  # noqa: BLE001
        logger.info("orthomosaic status lookup failed: %s", exc)
        url = None
    if url and state["state"] not in (QUEUED, RUNNING):
        state["state"] = COMPLETED
    if url:
        state["downloadUrl"] = url
    return state


def _semaphore() -> threading.BoundedSemaphore:
    global _slots
    with _guard:
        if _slots is None:
            _slots = threading.BoundedSemaphore(max(1, int(getattr(settings, "DVSA_ORTHOMOSAIC_MAX_JOBS", 1))))
        return _slots


def run_job(account_id, video_pk, source_url: str) -> Dict[str, Any]:
    """Generate and upload the orthomosaic for one video. Never raises; returns the final state."""
    from core.azure import blob as azure_blob  # noqa: PLC0415
    from core.azure import orthomosaic as ortho  # noqa: PLC0415

    key = (str(account_id), str(video_pk))
    video_path = work_dir = None
    persistent = bool(getattr(settings, "DVSA_ORTHOMOSAIC_WORK_DIR", ""))
    try:
        with _semaphore():
            _set_state(account_id, video_pk, RUNNING)
            opts = build_options()
            config = AzureEnvironmentConfig.from_settings()
            # Fail fast on a box without Docker/ODM, before downloading a large video.
            ortho.resolve_engine(opts["backend"], opts["odm_image"], opts["odm_run"], None)
            if persistent:
                # Stable paths so the script's verified frame/model caches can be reused on re-runs.
                work_dir = os.path.join(settings.DVSA_ORTHOMOSAIC_WORK_DIR, f"{account_id}-{video_pk}")
                os.makedirs(work_dir, exist_ok=True)
                video_path = os.path.join(work_dir, "source.mp4")
                if not os.path.isfile(video_path):
                    tmp = azure_blob.download_blob_to_temp(source_url)
                    shutil.move(tmp, video_path)
                work_dir = os.path.join(work_dir, "work")
            else:
                video_path = azure_blob.download_blob_to_temp(source_url)
                work_dir = tempfile.mkdtemp(prefix="dvsa-ortho-")
            jpeg, report = ortho.build_orthomosaic(video_path, work_dir, **opts)
            blob_name = ortho.orthomosaic_blob_name(account_id, video_pk)
            url = azure_blob.put_blob_and_sas(config, blob_name, jpeg)
            summary = report.get("output", {})
            logger.info("orthomosaic for video %s uploaded (%s)", video_pk, summary)
            return _set_state(account_id, video_pk, COMPLETED, downloadUrl=url,
                              blob=blob_name, output=summary)
    except Exception as exc:  # noqa: BLE001 - must never impact the rest of the system
        unavailable = type(exc).__name__ == "OrthomosaicUnavailable" or "Docker" in str(exc)
        logger.warning("orthomosaic for video %s %s: %s", video_pk,
                       "unavailable" if unavailable else "failed", exc)
        return _set_state(account_id, video_pk, FAILED, reason=str(exc)[:500], unavailable=unavailable)
    finally:
        for path, is_dir in ((None if persistent else video_path, False), (None if persistent else work_dir, True)):
            if path and os.path.exists(path):
                try:
                    shutil.rmtree(path, ignore_errors=True) if is_dir else os.remove(path)
                except OSError as exc:
                    logger.info("orthomosaic temp cleanup failed: %s", exc)
        with _guard:
            _active.discard(key)


def start_job(account_id, video_pk, source_url: Optional[str], *, force: bool = False,
              config: Optional[AzureEnvironmentConfig] = None, wait: bool = False) -> Dict[str, Any]:
    """Start (or reuse) the background job for a video; returns the current state immediately.

    Skips when the blob already exists (unless ``force``) or a job for this video is already
    running. With ``wait=True`` the job runs inline and the final state is returned.
    """
    if not source_url:
        return {"state": FAILED, "reason": "no source video url"}
    key = (str(account_id), str(video_pk))
    if not force:
        current = status(account_id, video_pk, config)
        if current["state"] in (COMPLETED, QUEUED, RUNNING):
            return current
    with _guard:
        if key in _active:
            return get_state(account_id, video_pk)
        _active.add(key)
    _set_state(account_id, video_pk, QUEUED)
    if wait:
        return run_job(account_id, video_pk, source_url)
    try:
        threading.Thread(target=run_job, args=(account_id, video_pk, source_url),
                         name=f"orthomosaic-{video_pk}", daemon=True).start()
    except Exception as exc:  # noqa: BLE001
        with _guard:
            _active.discard(key)
        return _set_state(account_id, video_pk, FAILED, reason=str(exc)[:500])
    return get_state(account_id, video_pk)
