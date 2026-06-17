"""Azure Blob Storage data-plane helpers (ported from ezvision).

Ports the blob/SAS plumbing from ``videos/myvideoindexer.py`` and the SAS-token
helpers from ``videos/analyzer_functions.py`` into config-driven functions. They
operate on the ``sadronevideo`` account's ``input``/``output`` containers.

``get_image_blob_url`` is a pure URL transform (no I/O) and is therefore always
available; the rest require ``azure-storage-blob`` and credentials and raise if
those are missing (callers handle the offline path).
"""

from __future__ import annotations

import datetime
import io
import logging
import os
import uuid
from urllib.parse import urlparse

from .config import AzureEnvironmentConfig

logger = logging.getLogger("apps.azure")


# --------------------------------------------------------------------------
# Pure URL helpers
# --------------------------------------------------------------------------
def get_image_blob_url(video_url, frame_number, folder="images", prefix="frame",
                       include_name=False, video_id=None):
    """Derive a frame's blob URL from the source video URL (verbatim port)."""
    parsed = urlparse(video_url)
    path_parts = parsed.path.split("/")
    blob_name = path_parts[-1].split(".")[0]
    container = path_parts[1] if len(path_parts) > 1 else ""
    blob_path = "/".join(path_parts[2:])
    blob_dir = "/".join(blob_path.split("/")[:-1])
    if blob_dir == "" or blob_dir is None:
        blob_dir = "output"
    new_path = f"{blob_dir}/{folder}"
    if video_id:
        new_path += "/" + str(video_id)
    if include_name:
        prefix += blob_name
    image_path = f"{new_path}/{prefix}{frame_number}.jpg"
    base_url = f"{parsed.scheme}://{parsed.netloc}/{container}/{image_path}"
    sas_token = parsed.query
    return f"{base_url}?{sas_token}" if sas_token else base_url


def get_sas_url_for_frame(sas_url_template, frame_number):
    """Substitute the frame number placeholder in a SAS URL template."""
    try:
        return sas_url_template.replace("frame(number)", f"frame{frame_number}")
    except Exception as exc:  # noqa: BLE001
        logger.info("get_sas_url_for_frame failed: %s", exc)
        return None


# --------------------------------------------------------------------------
# Authenticated blob operations
# --------------------------------------------------------------------------
def _account_url(config: AzureEnvironmentConfig) -> str:
    return f"https://{config.storage_account}.blob.core.windows.net"


def service_client(config: AzureEnvironmentConfig):
    """Return a ``BlobServiceClient`` for the configured account."""
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

    if config.storage_connection_string:
        return BlobServiceClient.from_connection_string(config.storage_connection_string)
    if not config.account_key:
        raise RuntimeError("AZURE_ACCOUNT_KEY (or connection string) required for blob ops")
    return BlobServiceClient(account_url=_account_url(config), credential=config.account_key)


def container_sas_url(config: AzureEnvironmentConfig, blob_url: str,
                      *, upload: bool = False, container: str = None) -> str:
    """Re-sign ``blob_url`` with a 1-hour container SAS (port of source token gen)."""
    from azure.storage.blob import BlobSasPermissions, generate_container_sas  # noqa: PLC0415

    container = container or config.input_container
    permission = BlobSasPermissions(read=True, list=True)
    if upload:
        permission = BlobSasPermissions(
            read=True, write=True, create=True, list=True, add=True,
            delete_previous_version=True,
        )
    sas_token = generate_container_sas(
        account_name=config.storage_account,
        container_name=container,
        account_key=config.account_key,
        permission=permission,
        expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1),
    )
    return blob_url.split("?")[0] + "?" + sas_token


def read_image_from_blob(sas_url: str):
    """Read an image from a blob SAS URL into an OpenCV ndarray (or None)."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import requests  # noqa: PLC0415

    resp = requests.get(sas_url, timeout=60)
    if resp.status_code == 200:
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None


def upload_image_to_blob(image_bytes, object_url: str) -> None:
    """Upload bytes to the blob addressed by ``object_url`` (SAS URL)."""
    from azure.storage.blob import BlobClient  # noqa: PLC0415

    try:
        BlobClient.from_blob_url(object_url).upload_blob(image_bytes, overwrite=True)
    except Exception as exc:  # noqa: BLE001
        logger.info("upload_image_to_blob failed: %s", exc)


def download_blob_to_stream(blob_client):
    return io.BytesIO(blob_client.download_blob().readall())


def copy_blob(source_sas_url: str, destination_sas_url: str, poll_interval: int = 2):
    """Server-side copy a blob and poll to completion. Returns final status."""
    import time  # noqa: PLC0415

    from azure.storage.blob import BlobClient  # noqa: PLC0415

    status = None
    try:
        dest = BlobClient.from_blob_url(destination_sas_url)
        props = dest.start_copy_from_url(source_sas_url)
        logger.info("Copy initiated: %s status=%s", props["copy_id"], props["copy_status"])
        while True:
            status = dest.get_blob_properties().copy.status
            if status in ("success", "failed", "aborted"):
                break
            time.sleep(poll_interval)
    except Exception as exc:  # noqa: BLE001
        logger.info("copy_blob failed: %s", exc)
    return status


def extract_and_upload_frames(config: AzureEnvironmentConfig, video_sas_url: str,
                              video_id=None) -> int:
    """Download a video, split into JPEG frames, and upload each to blob."""
    import cv2  # noqa: PLC0415
    from azure.storage.blob import BlobClient  # noqa: PLC0415

    video_blob_client = BlobClient.from_blob_url(video_sas_url)
    video_bytes = download_blob_to_stream(video_blob_client).getvalue()
    video_temp = os.path.join(os.getcwd(), f"temp_{uuid.uuid4()}.mp4")
    with open(video_temp, "wb") as f:
        f.write(video_bytes)
    vidcap = cv2.VideoCapture(video_temp)
    frame_number = 0
    try:
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            image_url = get_image_blob_url(video_sas_url, frame_number, video_id=video_id).strip('"')
            BlobClient.from_blob_url(image_url).upload_blob(buffer.tobytes(), overwrite=True)
            logger.info("Uploaded frame %s to %s", frame_number, image_url)
            frame_number += 1
    finally:
        vidcap.release()
        if os.path.exists(video_temp):
            try:
                os.remove(video_temp)
            except OSError as exc:
                logger.info("temp cleanup failed: %s", exc)
    return frame_number


def get_uploaded_frames(config: AzureEnvironmentConfig, video_sas_url: str,
                        account_id=None, video_id=None) -> int:
    """Count contiguous already-uploaded frames for a video (highest index)."""
    container = config.input_container
    try:
        svc = service_client(config)
    except Exception as exc:  # noqa: BLE001
        logger.info("get_uploaded_frames: %s", exc)
        return 0
    prefix = f"{_account_url(config)}/{container}/"
    frame_number = 0
    for frame_number in range(9999):
        try:
            image_url = get_image_blob_url(video_sas_url, frame_number, video_id=video_id).strip('"')
            blob_name = image_url.split("?")[0].replace(prefix, "")
            svc.get_blob_client(container=container, blob=blob_name).get_blob_properties()
        except Exception:  # noqa: BLE001 - first missing frame ends the scan
            break
    return frame_number


def get_destination_sas_url(config: AzureEnvironmentConfig, video_sas_url: str,
                            upload: bool = True) -> str:
    """Build the `<name>_indexed` destination SAS URL for the rendered video."""
    signed = container_sas_url(config, video_sas_url, upload=upload)
    parsed = urlparse(signed)
    blob_name = parsed.path.split("/")[-1].split(".")[0]
    return signed.replace(blob_name, blob_name + "_indexed")
