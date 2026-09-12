"""Frame extraction for the ``/extract-frames/`` direct tool endpoint.

Keeps all Azure / OpenCV I/O out of :class:`~apps.videos.views.FrameExtractAPIView`.
Given a source video SAS URL it returns a list of frame descriptors
(``frame_number``, ``blob_name``, ``sas_url``, optional ``t`` seconds) written
into the *existing* ingestion blob layout::

    {account_id}/images/{video_id}/frame{N}.jpg      # N contiguous from 0

Frame source is chosen in this order (see ``docs/direct_tool_endpoints.md``):

1. **existing** — frames already uploaded for this video (idempotent re-mint; no
   decode). This also makes the ingestion pipeline's ``get_uploaded_frames`` skip
   its full-frame dump.
2. **video_indexer** — salient key frames from Azure Video Indexer, when configured.
3. **stride** — sample one frame every ``stride`` seconds (default 10).

The blob layout is never modified: numbering stays contiguous ``frame0, frame1, …``
and timestamps live on the response / ``ImageEntity``, never in the blob name.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from core.azure.config import AzureEnvironmentConfig

logger = logging.getLogger("apps.videos")

Frame = Dict[str, object]  # {frame_number, blob_name, sas_url, t}
Sample = Tuple[Optional[float], bytes]  # (timestamp_seconds or None, jpeg bytes)


def extract_frames(config: AzureEnvironmentConfig, video_sas_url: str, *,
                   account_id: str, video_id: Optional[int] = None,
                   stride_seconds: float = 10.0) -> Tuple[str, List[Frame]]:
    """Resolve/produce the frame set for ``video_sas_url`` and return
    ``(source, frames)``. ``source`` is ``existing`` | ``video_indexer`` | ``stride``.
    """
    from core.azure import blob as azure_blob  # noqa: PLC0415

    # 1) Idempotent: reuse frames already present under this prefix (no decode).
    existing = azure_blob.get_uploaded_frames(
        config, video_sas_url, account_id=account_id, video_id=video_id)
    if existing > 0:
        frames = [
            _describe(config, video_sas_url, n, video_id, timestamp=None, upload=None)
            for n in range(existing)
        ]
        return "existing", frames

    # Need the video bytes for VI-sampling or stride-sampling.
    video_path = azure_blob.download_blob_to_temp(video_sas_url)
    try:
        source = "stride"
        samples: Optional[List[Sample]] = None
        try:
            from core.azure.video_indexer import VideoIndexerClient  # noqa: PLC0415

            vi = VideoIndexerClient(config)
            if vi.configured:
                samples = _video_indexer_samples(vi, video_sas_url, video_path)
                if samples:
                    source = "video_indexer"
        except Exception as exc:  # noqa: BLE001 - VI is best-effort; fall back
            logger.warning("video indexer sampling skipped: %s", exc)
            samples = None

        if not samples:
            samples = _stride_samples(video_path, stride_seconds)
            source = "stride"
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as exc:
                logger.info("extract-frames temp cleanup failed: %s", exc)

    frames = [
        _describe(config, video_sas_url, n, video_id, timestamp=ts, upload=jpg)
        for n, (ts, jpg) in enumerate(samples)
    ]
    return source, frames


def _describe(config: AzureEnvironmentConfig, video_sas_url: str, n: int,
              video_id: Optional[int], *, timestamp: Optional[float],
              upload: Optional[bytes]) -> Frame:
    """Upload one frame (when ``upload`` bytes are given) or just re-mint its SAS,
    and return the response descriptor."""
    from core.azure import blob as azure_blob  # noqa: PLC0415

    blob_name = azure_blob.image_blob_name(config, video_sas_url, n, video_id=video_id)
    if upload is None:
        sas_url = azure_blob.read_sas_for_blob(config, blob_name)
    else:
        sas_url = azure_blob.put_blob_and_sas(config, blob_name, upload)
    return {"frame_number": n, "blob_name": blob_name, "sas_url": sas_url, "t": timestamp}


def _stride_samples(video_path: str, stride_seconds: float) -> List[Sample]:
    """Sample one JPEG every ``stride_seconds`` seconds, numbered 0..K-1.

    Uses ``CAP_PROP_POS_MSEC`` seeking when duration is known; otherwise falls
    back to a frame stride derived from FPS so numbering stays contiguous.
    """
    import cv2  # noqa: PLC0415

    if stride_seconds <= 0:
        raise ValueError("stride must be > 0 seconds")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        samples: List[Sample] = []

        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            n = 0
            while True:
                t = n * stride_seconds
                if t > duration:
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    break
                ok2, buf = cv2.imencode(".jpg", frame)
                if ok2:
                    samples.append((round(t, 3), buf.tobytes()))
                n += 1
        else:
            # Unknown metadata: sample every ~stride*fps frames sequentially.
            fps_eff = fps if fps > 0 else 30.0
            step = max(1, int(round(fps_eff * stride_seconds)))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    ok2, buf = cv2.imencode(".jpg", frame)
                    if ok2:
                        samples.append((round(idx / fps_eff, 3), buf.tobytes()))
                idx += 1
        return samples
    finally:
        cap.release()


def _video_indexer_samples(vi, video_sas_url: str, video_path: str) -> Optional[List[Sample]]:
    """Sample the salient key-frame timestamps reported by Azure Video Indexer.

    Returns ``None`` (caller falls back to stride) if indexing/insights are
    unavailable or yield no key frames.
    """
    import cv2  # noqa: PLC0415

    token = vi.get_access_token()
    if not token:
        return None
    video_id = vi.get_uploaded_video_id(token, video_url=video_sas_url)
    if not video_id:
        return None
    insights = vi.get_video_insights(token, video_id)
    if not insights:
        return None
    segments = vi.get_selected_segments(insights, 10)
    times = sorted({_vi_seconds(start) for start, _ in segments if start is not None})
    times = [t for t in times if t is not None]
    if not times:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        samples: List[Sample] = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                samples.append((round(t, 3), buf.tobytes()))
        return samples or None
    finally:
        cap.release()


def _vi_seconds(ts: str) -> Optional[float]:
    """Parse a Video Indexer timestamp (``H:MM:SS(.ffffff)``) into seconds."""
    try:
        parts = str(ts).split(":")
        parts = [float(p) for p in parts]
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + p
        return round(seconds, 3)
    except (ValueError, TypeError):
        return None
