"""Sample test fixtures for the planned P7 photogrammetry / multi-tour orthomosaic provider.

Nothing here touches production code. It gives the future provider tests three things
with *known ground truth*, so a result can be asserted against, not just "it ran":

1. ``make_ground`` -- a feature-rich synthetic top-down scene (stands in for the true
   orthomosaic) with a local metric frame and a geographic origin.
2. ``make_tour`` -- a lawnmower flight over part of that scene: overlapping nadir frames
   (pixel crops of the ground), each with its true pose, video offset, lat/lon/alt, plus a
   DJI-style ``.SRT`` telemetry sidecar and ``frame{N}.jpg`` files. Two tours over
   overlapping parts of the scene give the "one orthomosaic from several tours" input.
3. ``FakeNodeODM`` -- an in-process HTTP server that speaks the small slice of the NodeODM
   REST API the provider needs (``/task/new``, ``/task/<id>/info``,
   ``/task/<id>/download/<asset>``, ``/task/cancel``, ``/task/remove``) with scripted
   status progression and failure injection, and serves the ground truth back as the
   "orthophoto" and the true camera poses as ``shots.geojson``.

pytest fixtures: ``ground``, ``two_tours``, ``no_gps_tour``, ``fake_nodeodm``.
Import them into a test module (``from tests.photogrammetry_fixtures import *``) or
register ``tests.photogrammetry_fixtures`` via ``pytest_plugins`` in a conftest.

NodeODM status codes and endpoint shapes follow the public NodeODM API as remembered at
planning time -- verify against the NodeODM version actually deployed (plan 8.3).
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

# --------------------------------------------------------------------------- constants
ORIGIN_LAT, ORIGIN_LON = 47.6000, -122.3000        # geographic position of the ground's top-left
METRES_PER_DEG_LAT = 110_540.0                       # same equirectangular constants as survey_corners.py
METRES_PER_DEG_LON_AT_EQUATOR = 111_320.0

# NodeODM task status codes.
QUEUED, RUNNING, FAILED, COMPLETED, CANCELED = 10, 20, 30, 40, 50
_STATUS_CODE = {"queued": QUEUED, "running": RUNNING, "failed": FAILED,
                "completed": COMPLETED, "cancelled": CANCELED, "canceled": CANCELED}


# --------------------------------------------------------------------------- ground truth
@dataclass(frozen=True)
class Ground:
    """A synthetic orthomosaic: ``image`` is RGB uint8, north-up, ``gsd_m`` metres/pixel."""

    image: np.ndarray
    gsd_m: float
    origin_lat: float = ORIGIN_LAT
    origin_lon: float = ORIGIN_LON

    @property
    def height_px(self) -> int:
        return self.image.shape[0]

    @property
    def width_px(self) -> int:
        return self.image.shape[1]

    def px_to_local_m(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Pixel (x right, y down) -> local (east, north) metres from the top-left origin."""
        return x_px * self.gsd_m, (self.height_px - y_px) * self.gsd_m

    def local_m_to_latlon(self, east_m: float, north_m: float) -> Tuple[float, float]:
        # Local frame origin is the *bottom-left* corner; the top-left sits height*gsd north of it.
        top_north = self.height_px * self.gsd_m
        lat = self.origin_lat - (top_north - north_m) / METRES_PER_DEG_LAT
        lon = self.origin_lon + east_m / (METRES_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(self.origin_lat)))
        return lat, lon

    def geotransform(self) -> Dict[str, float]:
        """Ground-truth georeference of the full ground image (top-left origin, deg/px)."""
        lat_per_px = self.gsd_m / METRES_PER_DEG_LAT
        lon_per_px = self.gsd_m / (METRES_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(self.origin_lat)))
        return {"origin_lat": self.origin_lat, "origin_lon": self.origin_lon,
                "lat_per_px": -lat_per_px, "lon_per_px": lon_per_px, "gsd_m": self.gsd_m}


def make_ground(width_px: int = 1600, height_px: int = 1200, gsd_m: float = 0.10, seed: int = 7) -> Ground:
    """Deterministic, texture-rich top-down scene (plenty of corners for feature matching)."""
    import cv2

    rng = np.random.default_rng(seed)
    base = rng.integers(70, 130, size=(height_px, width_px, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(base, (0, 0), 6)
    for _ in range(900):                                           # buildings / patches
        x, y = int(rng.integers(0, width_px)), int(rng.integers(0, height_px))
        w, h = int(rng.integers(8, 70)), int(rng.integers(8, 70))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=-1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), thickness=1)
    for _ in range(300):                                           # trees / vehicles
        c = (int(rng.integers(0, width_px)), int(rng.integers(0, height_px)))
        cv2.circle(img, c, int(rng.integers(2, 12)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    for _ in range(80):                                            # roads / tracks
        p1 = (int(rng.integers(0, width_px)), int(rng.integers(0, height_px)))
        p2 = (int(rng.integers(0, width_px)), int(rng.integers(0, height_px)))
        cv2.line(img, p1, p2, (200, 200, 190), int(rng.integers(1, 4)))
    return Ground(image=img, gsd_m=gsd_m)


# --------------------------------------------------------------------------- tours
@dataclass
class FrameFixture:
    frame_id: str
    index: int                  # 0-based within the tour; file is frame{index + 1}.jpg
    video_offset_us: int
    bbox_px: Tuple[int, int, int, int]       # x0, y0, w, h on the ground image
    east_m: float
    north_m: float
    alt_m: float
    lat: float
    lon: float
    yaw_deg: float
    image: np.ndarray = field(repr=False)

    @property
    def filename(self) -> str:
        return f"frame{self.index + 1}.jpg"


@dataclass
class TourFixture:
    name: str
    frames: List[FrameFixture]
    ground: Ground
    has_gps: bool = True
    fps: float = 2.0

    def srt_text(self, start: str = "2026-10-01 10:00:00") -> str:
        """DJI-style SRT sidecar; parseable by ``core.azure.survey_corners.track_from_srt``."""
        step_ms = int(1000 / self.fps)
        out = []
        for f in self.frames:
            t0, t1 = f.video_offset_us // 1000, f.video_offset_us // 1000 + step_ms
            out.append(
                f"{f.index + 1}\n{_srt_ts(t0)} --> {_srt_ts(t1)}\n"
                f'<font size="28">FrameCnt: {f.index + 1}, DiffTime: {step_ms}ms\n'
                f"{start}.{t0 % 1000:03d}\n"
                f"[iso: 100] [shutter: 1/1000.0] [fnum: 280] [ev: 0] "
                f"[latitude: {f.lat:.7f}] [longitude: {f.lon:.7f}] "
                f"[rel_alt: {f.alt_m - 40:.3f} abs_alt: {f.alt_m:.3f}] [ct: 5500] </font>\n"
            )
        return "\n".join(out)

    def manifest(self) -> List[dict]:
        """Canonical sampled-frame manifest: join results by ``frame_id`` + offset, never row order."""
        return [{"frame_id": f.frame_id, "filename": f.filename, "video_offset_us": f.video_offset_us,
                 "lat": f.lat if self.has_gps else None, "lon": f.lon if self.has_gps else None,
                 "alt_m": f.alt_m if self.has_gps else None} for f in self.frames]

    def write(self, directory: Path) -> Dict[str, Path]:
        """Write ``frame{N}.jpg`` (+ ``<tour>.SRT`` and ``manifest.json`` when it has GPS)."""
        import cv2

        directory.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}
        for f in self.frames:
            path = directory / f.filename
            cv2.imwrite(str(path), cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            paths[f.frame_id] = path
        (directory / "manifest.json").write_text(json.dumps(self.manifest(), indent=2), encoding="utf-8")
        if self.has_gps:
            (directory / f"{self.name}.SRT").write_text(self.srt_text(), encoding="utf-8")
        return paths


def _srt_ts(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_tour(
    ground: Ground,
    name: str,
    *,
    rows: Tuple[int, int] = (0, 10**6),      # which flight lines (row indices) this tour covers
    frame_px: Tuple[int, int] = (480, 360),  # frame width, height in ground pixels
    forward_overlap: float = 0.75,
    side_overlap: float = 0.65,
    agl_m: float = 80.0,
    brightness: float = 1.0,                 # per-tour exposure/lighting difference
    has_gps: bool = True,
    fps: float = 2.0,
) -> TourFixture:
    """Lawnmower pass over ``ground``: nadir frames cropped from the ground image, with true poses."""
    fw, fh = frame_px
    step_x = max(1, int(fw * (1 - forward_overlap)))
    step_y = max(1, int(fh * (1 - side_overlap)))
    xs = list(range(0, ground.width_px - fw + 1, step_x))
    ys = list(range(0, ground.height_px - fh + 1, step_y))[rows[0]:rows[1]]
    frames: List[FrameFixture] = []
    for r, y in enumerate(ys):
        row_xs = xs if r % 2 == 0 else xs[::-1]          # boustrophedon
        for x in row_xs:
            i = len(frames)
            crop = ground.image[y:y + fh, x:x + fw]
            if brightness != 1.0:
                crop = np.clip(crop.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
            east, north = ground.px_to_local_m(x + fw / 2, y + fh / 2)
            lat, lon = ground.local_m_to_latlon(east, north)
            frames.append(FrameFixture(
                frame_id=f"{name}-frame-{i:05d}", index=i, video_offset_us=int(i * 1e6 / fps),
                bbox_px=(x, y, fw, fh), east_m=east, north_m=north, alt_m=agl_m + 40.0,
                lat=lat, lon=lon, yaw_deg=0.0 if r % 2 == 0 else 180.0, image=crop.copy()))
    return TourFixture(name=name, frames=frames, ground=ground, has_gps=has_gps, fps=fps)


def union_bbox_px(tours: Sequence[TourFixture]) -> Tuple[int, int, int, int]:
    """Ground-pixel rectangle covered by any frame of any tour (x0, y0, w, h)."""
    boxes = [f.bbox_px for t in tours for f in t.frames]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def ground_truth_orthomosaic(ground: Ground, tours: Sequence[TourFixture]) -> np.ndarray:
    """What a perfect reconstruction of these tours would produce (RGB, north-up)."""
    x0, y0, w, h = union_bbox_px(tours)
    return ground.image[y0:y0 + h, x0:x0 + w].copy()


def frame_overlap_fraction(a: FrameFixture, b: FrameFixture) -> float:
    """Fraction of ``a`` covered by ``b`` (0..1) -- used to assert the configured overlaps."""
    ax, ay, aw, ah = a.bbox_px
    bx, by, bw, bh = b.bbox_px
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (ix * iy) / float(aw * ah)


def shots_geojson(tours: Sequence[TourFixture]) -> dict:
    """True camera poses in the shape ODM's ``shots.geojson`` uses (Point + filename/altitude)."""
    feats = []
    for t in tours:
        for f in t.frames:
            feats.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [f.lon, f.lat, f.alt_m]},
                          "properties": {"filename": f"{t.name}_{f.filename}", "frame_id": f.frame_id,
                                         "translation": [f.east_m, f.north_m, f.alt_m], "yaw_deg": f.yaw_deg}})
    return {"type": "FeatureCollection", "features": feats}


# --------------------------------------------------------------------------- multipart helper
def encode_multipart(fields: Dict[str, str], files: Sequence[Tuple[str, str, bytes]]) -> Tuple[bytes, str]:
    """Build a multipart/form-data body: ``files`` are (field, filename, data). Returns (body, content_type)."""
    boundary = uuid.uuid4().hex
    parts: List[bytes] = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for field_name, filename, data in files:
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"; '
                      f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
                     + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _parse_multipart(body: bytes, content_type: str) -> Tuple[Dict[str, str], List[str]]:
    boundary = content_type.split("boundary=")[1].strip().encode()
    fields: Dict[str, str] = {}
    filenames: List[str] = []
    for part in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        head_s = head.decode("utf-8", "ignore")
        name = head_s.split('name="')[1].split('"')[0] if 'name="' in head_s else ""
        if 'filename="' in head_s:
            filenames.append(head_s.split('filename="')[1].split('"')[0])
        else:
            fields[name] = data.rstrip(b"\r\n").decode("utf-8", "ignore")
    return fields, filenames


# --------------------------------------------------------------------------- fake NodeODM
@dataclass
class _Task:
    uuid: str
    name: str
    options: list
    images: List[str]
    script: List[str]
    polls: int = 0
    cancelled: bool = False
    removed: bool = False

    def status(self) -> str:
        if self.cancelled:
            return "cancelled"
        state = self.script[min(self.polls, len(self.script) - 1)]
        self.polls += 1
        return state


class FakeNodeODM:
    """Scripted stand-in for a NodeODM server (start/stop with ``with`` or the pytest fixture).

    ``script``       status progression returned by successive ``/info`` polls
                     (default queued -> running -> running -> completed).
    ``fail_message`` when the script reaches ``failed``, ``info.status.errorMessage`` carries this.
    ``calls``        ordered log of ``(method, path)`` for asserting the provider's behaviour.
    ``tasks``        uuid -> task, to inspect submitted image names and options.
    """

    def __init__(self, ground: Ground, tours: Sequence[TourFixture],
                 script: Sequence[str] = ("queued", "running", "running", "completed"),
                 fail_message: str = "Cannot process dataset") -> None:
        self.ground, self.tours = ground, list(tours)
        self.script, self.fail_message = list(script), fail_message
        self.calls: List[Tuple[str, str]] = []
        self.tasks: Dict[str, _Task] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle
    @property
    def url(self) -> str:
        assert self._server, "server not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> "FakeNodeODM":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code: int, payload, content_type: str = "application/json") -> None:
                data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802
                outer.calls.append(("POST", self.path.split("?")[0]))
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                path = self.path.split("?")[0]
                if path == "/task/new":
                    fields, files = _parse_multipart(body, self.headers["Content-Type"])
                    task = _Task(uuid=str(uuid.uuid4()), name=fields.get("name", ""),
                                 options=json.loads(fields.get("options", "[]")), images=files,
                                 script=list(outer.script))
                    outer.tasks[task.uuid] = task
                    return self._send(200, {"uuid": task.uuid})
                if path in ("/task/cancel", "/task/remove"):
                    fields, _ = _parse_multipart(body, self.headers["Content-Type"]) \
                        if "multipart" in (self.headers.get("Content-Type") or "") else (json.loads(body or b"{}"), [])
                    task = outer.tasks.get(fields.get("uuid", ""))
                    if not task:
                        return self._send(200, {"success": False, "error": "Invalid task id"})
                    task.cancelled = task.cancelled or path.endswith("cancel")
                    task.removed = task.removed or path.endswith("remove")
                    return self._send(200, {"success": True})
                self._send(404, {"error": "not found"})

            def do_GET(self):  # noqa: N802
                path = self.path.split("?")[0]
                outer.calls.append(("GET", path))
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[0] == "task":
                    task = outer.tasks.get(parts[1])
                    if not task or task.removed:
                        return self._send(404, {"error": "Invalid task id"})
                    if parts[2] == "info":
                        state = task.status()
                        status = {"code": _STATUS_CODE[state]}
                        if state == "failed":
                            status["errorMessage"] = outer.fail_message
                        return self._send(200, {
                            "uuid": task.uuid, "name": task.name, "status": status,
                            "progress": {"queued": 0, "running": 50, "completed": 100}.get(state, 0),
                            "imagesCount": len(task.images), "options": task.options,
                            "processingTime": 1234 if state == "completed" else -1})
                    if parts[2] == "download" and len(parts) == 4:
                        return self._download(task, parts[3])
                self._send(404, {"error": "not found"})

            def _download(self, task: _Task, asset: str) -> None:
                if task.script[-1] != "completed" or task.polls == 0:
                    return self._send(404, {"error": "Task not completed"})
                if asset == "orthophoto.tif":
                    import cv2

                    ok, png = cv2.imencode(".png", cv2.cvtColor(
                        ground_truth_orthomosaic(outer.ground, outer.tours), cv2.COLOR_RGB2BGR))
                    # PNG bytes under the .tif name: enough for plumbing tests. Tests that need a
                    # real GeoTIFF/CRS should write one with rasterio from ground_truth_orthomosaic().
                    return self._send(200, png.tobytes(), "image/png")
                if asset == "shots.geojson":
                    return self._send(200, shots_geojson(outer.tours))
                if asset == "report.json":
                    return self._send(200, {"processing_statistics": {"gsd": outer.ground.gsd_m * 100},
                                            "reconstruction_statistics": {"reconstructed_images_count": len(task.images)}})
                self._send(404, {"error": f"unknown asset {asset}"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "FakeNodeODM":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# --------------------------------------------------------------------------- pytest fixtures
@pytest.fixture(scope="session")
def ground() -> Ground:
    return make_ground()


@pytest.fixture(scope="session")
def two_tours(ground: Ground) -> Tuple[TourFixture, TourFixture]:
    """Tour A covers the first flight lines, tour B the later ones; they share the middle rows
    (different brightness, as two flights at different times would)."""
    a = make_tour(ground, "tourA", rows=(0, 3), brightness=1.0)
    b = make_tour(ground, "tourB", rows=(2, 6), brightness=0.9)
    return a, b


@pytest.fixture(scope="session")
def no_gps_tour(ground: Ground) -> TourFixture:
    """Frames with no telemetry: the provider must treat the result as unreferenced."""
    return make_tour(ground, "tourNoGps", rows=(0, 2), has_gps=False)


@pytest.fixture
def fake_nodeodm(ground: Ground, two_tours):
    with FakeNodeODM(ground, two_tours) as server:
        yield server
