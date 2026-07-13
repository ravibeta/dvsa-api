"""Adapter layer: pluggable IO + inference for the drone-video pipeline.

Defines four typed interfaces — fetch video, extract frames, run inference, store
output — plus offline, deterministic reference implementations:

* :class:`LocalVideoFetcher` — resolves ``file://`` / path URIs to a local file.
* :class:`SyntheticFrameExtractor` — reads frames from a synthetic ``.json`` video
  manifest (see ``agent_kits/test_assets/generate_synthetic_video.py``); falls back
  to OpenCV for real videos *only if it is installed* (lazy import).
* :class:`MockInferenceAdapter` — deterministic detections, ideal for tests.
* :class:`LocalFileStorageAdapter` — persists :class:`RunOutput` JSON to disk.

Third parties implement the same Protocols to plug in S3/GCS storage, real models,
etc. — see ``agent_kits/README.md`` extension guide.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:  # Protocol is stdlib on 3.8+; fall back for older/local interpreters.
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover - only on Python < 3.8
    try:
        from typing_extensions import Protocol, runtime_checkable  # type: ignore
    except ImportError:  # last resort: degrade Protocol to a plain base
        class Protocol:  # type: ignore
            ...

        def runtime_checkable(cls):  # type: ignore
            return cls

from .schemas import Detection, Provenance, RunOutput, TimeWindow


@dataclass
class Frame:
    """A single extracted frame (metadata + optional ground-truth objects)."""

    index: int
    timestamp: float
    width: int = 0
    height: int = 0
    # Optional synthetic ground-truth objects used by the mock inference adapter.
    objects: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Interfaces (Protocols) — extension points for third parties.
# --------------------------------------------------------------------------- #
@runtime_checkable
class VideoFetcher(Protocol):
    def fetch_video(self, video_uri: str) -> str:
        """Resolve ``video_uri`` to a readable local path."""


@runtime_checkable
class FrameExtractor(Protocol):
    def extract_frames(
        self, video_path: str, *, fps: float = 1.0,
        time_window: Optional[TimeWindow] = None,
    ) -> List[Frame]:
        """Extract frames at ``fps`` within an optional time window."""


@runtime_checkable
class InferenceAdapter(Protocol):
    model_version: str

    def run_inference_on_frames(self, frames: Sequence[Frame]) -> List[Detection]:
        """Run detection over frames, returning :class:`Detection` objects."""


@runtime_checkable
class StorageAdapter(Protocol):
    def store_output(self, output: RunOutput, dest: str) -> str:
        """Persist ``output`` and return its storage URI."""


# --------------------------------------------------------------------------- #
# Reference implementations (offline, deterministic).
# --------------------------------------------------------------------------- #
class LocalVideoFetcher:
    """Resolve ``file://`` and bare path URIs to a local filesystem path."""

    def fetch_video(self, video_uri: str) -> str:
        path = video_uri[len("file://"):] if video_uri.startswith("file://") else video_uri
        if video_uri.split("://", 1)[0] in ("s3", "gs", "gcs", "http", "https") \
                and "://" in video_uri:
            raise NotImplementedError(
                f"Remote scheme not supported offline: {video_uri!r}. "
                "Provide a custom VideoFetcher (see README extension guide).")
        if not os.path.exists(path):
            raise FileNotFoundError(f"video not found: {path}")
        return path


class SyntheticFrameExtractor:
    """Read frames from a synthetic ``.json`` video manifest.

    Manifest shape::

        {"fps": 5, "width": 640, "height": 480,
         "frames": [{"index": 0, "timestamp": 0.0,
                     "objects": [{"bbox": [x,y,w,h], "class_name": "car",
                                  "confidence": 0.9}]}]}

    For real (binary) videos, OpenCV is imported lazily *iff* installed; otherwise a
    clear error asks the caller to supply a synthetic manifest or custom extractor.
    """

    def extract_frames(
        self, video_path: str, *, fps: float = 1.0,
        time_window: Optional[TimeWindow] = None,
    ) -> List[Frame]:
        if video_path.endswith(".json"):
            return self._from_manifest(video_path, time_window)
        return self._from_video(video_path, fps=fps, time_window=time_window)

    @staticmethod
    def _in_window(ts: float, window: Optional[TimeWindow]) -> bool:
        if window is None:
            return True
        start = _to_seconds(window.start)
        end = _to_seconds(window.end)
        if start is not None and ts < start:
            return False
        if end is not None and ts > end:
            return False
        return True

    def _from_manifest(
        self, path: str, window: Optional[TimeWindow],
    ) -> List[Frame]:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        width = int(manifest.get("width", 0))
        height = int(manifest.get("height", 0))
        frames: List[Frame] = []
        for raw in manifest.get("frames", []):
            ts = float(raw.get("timestamp", raw.get("index", 0)))
            if not self._in_window(ts, window):
                continue
            frames.append(Frame(
                index=int(raw.get("index", len(frames))),
                timestamp=ts,
                width=int(raw.get("width", width)),
                height=int(raw.get("height", height)),
                objects=list(raw.get("objects", [])),
            ))
        return sorted(frames, key=lambda f: (f.timestamp, f.index))

    def _from_video(
        self, path: str, *, fps: float, window: Optional[TimeWindow] = None,
        time_window: Optional[TimeWindow] = None,
    ) -> List[Frame]:
        window = window or time_window
        try:
            import cv2  # noqa: PLC0415 - optional heavy dep, imported only if present
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OpenCV not available and video is not a synthetic .json manifest. "
                "Install opencv-python or provide a manifest / custom FrameExtractor."
            ) from exc
        cap = cv2.VideoCapture(path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native_fps / fps))) if fps > 0 else 1
        frames: List[Frame] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts = idx / native_fps
                if self._in_window(ts, window):
                    h, w = frame.shape[:2]
                    frames.append(Frame(index=idx, timestamp=round(ts, 6),
                                        width=int(w), height=int(h)))
            idx += 1
        cap.release()
        return frames


class MockInferenceAdapter:
    """Deterministic detector for tests and offline runs.

    Emits one :class:`Detection` per synthetic ground-truth object on each frame.
    Fully reproducible: identical frames always yield identical detections.
    """

    def __init__(self, model_version: str = "mock-detector-1.0.0") -> None:
        self.model_version = model_version

    def run_inference_on_frames(self, frames: Sequence[Frame]) -> List[Detection]:
        detections: List[Detection] = []
        for frame in frames:
            for obj in frame.objects:
                detections.append(Detection(
                    timestamp=float(frame.timestamp),
                    bbox=[float(x) for x in obj.get("bbox", [0, 0, 0, 0])],
                    class_name=str(obj.get("class_name", "object")),
                    confidence=float(obj.get("confidence", 0.5)),
                    model_version=self.model_version,
                    provenance=Provenance(source="MockInferenceAdapter",
                                          method="synthetic-groundtruth"),
                ))
        return detections


class LocalFileStorageAdapter:
    """Persist a :class:`RunOutput` as pretty JSON on the local filesystem."""

    def store_output(self, output: RunOutput, dest: str) -> str:
        path = dest[len("file://"):] if dest.startswith("file://") else dest
        if os.path.isdir(path) or dest.endswith(("/", "\\")):
            path = os.path.join(path, f"{output.run_id}.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(output.model_dump(), fh, indent=2, sort_keys=True, default=str)
        return f"file://{os.path.abspath(path)}"


def _to_seconds(value: Optional[str]) -> Optional[float]:
    """Best-effort parse of an ISO timestamp or numeric-seconds string."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime  # noqa: PLC0415
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# DVSA bridge adapters — optionally leverage the existing DRF views.
# --------------------------------------------------------------------------- #
# Rather than the offline reference codepaths above, these adapters invoke the
# production ``apps.videos.views`` **in-process** so the kits reuse their built-in
# smarts: ``VideoUploadAPIView`` (Azure blob upload + SAS + VideoEntity registration
# + the signal-driven ingestion/indexing pipeline) and ``ChatAPIView`` (agentic RAG
# synthesis over indexed frames via ``env.ask``). Django/DRF/Azure are imported
# lazily, so importing ``agent_kits`` stays dependency-free and offline-safe; these
# adapters are opt-in and only pull those deps when actually used.


def _default_view_caller(view_cls, method, data, *, user=None, path="/"):
    """Invoke a DRF ``APIView`` in-process; return ``(status_code, response_data)``.

    Uses DRF's request factory so the view runs with its real parsing, permissions
    and response handling — i.e. we genuinely exercise the existing view, not a copy.
    """
    from rest_framework.test import (  # noqa: PLC0415
        APIRequestFactory, force_authenticate)
    factory = APIRequestFactory()
    # A file-like value means a multipart upload; otherwise send JSON.
    fmt = "multipart" if any(hasattr(v, "read") for v in data.values()) else "json"
    request = getattr(factory, method.lower())(path, data, format=fmt)
    if user is not None:
        force_authenticate(request, user=user)
    response = view_cls.as_view()(request)
    return response.status_code, getattr(response, "data", None)


def _load_video_upload_view():
    from apps.videos.views import VideoUploadAPIView  # noqa: PLC0415
    return VideoUploadAPIView


def _load_chat_view():
    from apps.videos.views import ChatAPIView  # noqa: PLC0415
    return ChatAPIView


class DvsaVideoUploadAdapter:
    """``VideoFetcher`` that ingests through the existing ``VideoUploadAPIView``.

    :meth:`ingest_video` uploads the local file via the production view, reusing its
    Azure blob upload, SAS generation, ``VideoEntity`` registration and the
    signal-driven indexing pipeline, and returns the created VideoEntity payload.
    :meth:`fetch_video` resolves the local path for downstream frame extraction and,
    when ``ingest_on_fetch`` is set, ingests first so a single pipeline run both
    indexes (Azure) and analyses (local) the video.

    ``account_id``/``user`` may be passed explicitly or taken from ``DVSA_ACCOUNT_ID``.
    ``view``/``view_caller`` are injection seams for testing without Django/DRF.
    """

    def __init__(self, *, account_id=None, user=None, view=None, view_caller=None,
                 ingest_on_fetch=False, path="/api/videos/upload-video/"):
        self.account_id = account_id or os.environ.get("DVSA_ACCOUNT_ID")
        self.user = user
        self._view = view
        self._view_caller = view_caller or _default_view_caller
        self.ingest_on_fetch = ingest_on_fetch
        self.path = path
        self._local = LocalVideoFetcher()

    def ingest_video(self, video_uri, *, account_id=None, user=None):
        """Upload + register + index ``video_uri`` via ``VideoUploadAPIView``."""
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError(
                "DvsaVideoUploadAdapter requires an account_id (arg or DVSA_ACCOUNT_ID)")
        local_path = self._local.fetch_video(video_uri)
        view_cls = self._view or _load_video_upload_view()
        with open(local_path, "rb") as handle:
            status_code, data = self._view_caller(
                view_cls, "post",
                {"file": handle, "account_id": account_id},
                user=user or self.user, path=self.path)
        if status_code and status_code >= 400:
            raise RuntimeError(f"VideoUploadAPIView failed ({status_code}): {data}")
        return {"status_code": status_code, "video_entity": data}

    def fetch_video(self, video_uri):
        if self.ingest_on_fetch:
            self.ingest_video(video_uri)
        return self._local.fetch_video(video_uri)


class DvsaChatAnalyzer:
    """Answer a query over an account's indexed frames via ``ChatAPIView``.

    Leverages the production chat view's agentic synthesis (``env.ask``) instead of
    reimplementing retrieval. Chat returns synthesized *text* (not bounding boxes), so
    this is an analyzer — plug it into :func:`run_pipeline` via ``analyzer=``/``query=``
    to attach the answer to the run's summary.
    """

    def __init__(self, *, account_id=None, user=None, view=None, view_caller=None,
                 path="/api/videos/chat/"):
        self.account_id = account_id or os.environ.get("DVSA_ACCOUNT_ID")
        self.user = user
        self._view = view
        self._view_caller = view_caller or _default_view_caller
        self.path = path

    def ask(self, query, *, account_id=None, user=None):
        """Return ``{'answer': text, ...}`` synthesised by ``ChatAPIView``."""
        account_id = account_id or self.account_id
        if not account_id:
            raise ValueError(
                "DvsaChatAnalyzer requires an account_id (arg or DVSA_ACCOUNT_ID)")
        if not query:
            raise ValueError("query is required")
        view_cls = self._view or _load_chat_view()
        status_code, data = self._view_caller(
            view_cls, "put",
            {"account_id": account_id, "query": query},
            user=user or self.user, path=self.path)
        if status_code and status_code >= 400:
            raise RuntimeError(f"ChatAPIView failed ({status_code}): {data}")
        answer = data.get("text") if isinstance(data, dict) else None
        return {"status_code": status_code, "answer": answer, "raw": data}


__all__ = [
    "Frame",
    "VideoFetcher",
    "FrameExtractor",
    "InferenceAdapter",
    "StorageAdapter",
    "LocalVideoFetcher",
    "SyntheticFrameExtractor",
    "MockInferenceAdapter",
    "LocalFileStorageAdapter",
    "DvsaVideoUploadAdapter",
    "DvsaChatAnalyzer",
]
