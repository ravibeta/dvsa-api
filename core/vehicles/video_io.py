"""Video probing, timestamp-correct sequential decoding, frame selection and H.264 MP4 writing.

Timestamps come from the container (``CAP_PROP_POS_MSEC`` after each ``read()``), never from
``index / fps``: drone and phone recordings are often variable-frame-rate. Decoding is strictly
sequential because random seeking in H.264 with B-frames is unreliable in OpenCV.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from .ffmpeg import find_ffmpeg, find_ffprobe

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    frame_count: int
    duration_s: float
    fps_nominal: float
    fps_average: float
    codec: Optional[str]
    variable_frame_rate: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filename"] = os.path.basename(self.path)
        del d["path"]
        return d


def _ffprobe(path: str) -> Optional[dict]:
    exe = find_ffprobe()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
             "stream=width,height,codec_name,r_frame_rate,avg_frame_rate,nb_read_packets,duration",
             "-of", "json", path],
            capture_output=True, text=True, check=True, timeout=60)
        return json.loads(out.stdout)["streams"][0]
    except (subprocess.SubprocessError, OSError, KeyError, IndexError, json.JSONDecodeError):
        return None


def _rate(value: Optional[str]) -> float:
    try:
        return float(Fraction(value)) if value else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: str) -> VideoInfo:
    """Resolution, frame count, duration and frame rates (ffprobe when available, else OpenCV)."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_cv, fps_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    s = _ffprobe(path)
    if s:
        n = int(s.get("nb_read_packets") or n_cv)
        nominal = _rate(s.get("r_frame_rate")) or fps_cv
        dur = float(s.get("duration") or (n / nominal if nominal else 0.0))
        codec = s.get("codec_name")
        w, h = int(s.get("width") or w), int(s.get("height") or h)
    else:
        n, nominal, codec = n_cv, fps_cv, None
        dur = n / fps_cv if fps_cv else 0.0
    nominal = nominal or 30.0
    avg = n / dur if dur > 0 else nominal
    return VideoInfo(path, w, h, n, round(dur, 4), round(nominal, 4), round(avg, 4), codec,
                     variable_frame_rate=abs(avg - nominal) > 0.01 * max(nominal, 1e-9))


def iter_frames(path: str, start_s: float = 0.0) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield ``(index, timestamp_s, frame_bgr)`` for every frame at or after ``start_s``.

    Frames before ``start_s`` are only grabbed (no BGR conversion), still sequentially.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        idx = 0
        while True:
            if not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t >= start_s - 1e-9:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield idx, t, frame
            idx += 1
    finally:
        cap.release()


class FrameSelector:
    """Pick detection frames by timestamp: one at least ``1/target_fps`` s after the previous pick.

    ``target_fps <= 0`` selects every frame. Half a source frame of tolerance keeps e.g. a 30 fps
    source at target 15 fps on exactly every second frame; a frame after a gap is always selected.
    """

    def __init__(self, target_fps: float, source_fps: float):
        self.interval = 1.0 / target_fps if target_fps and target_fps > 0 else 0.0
        self.tol = 0.5 / source_fps if source_fps and source_fps > 0 else 0.0
        self.last: Optional[float] = None

    def __call__(self, t: float) -> bool:
        if self.last is None or self.interval == 0.0 or t - self.last >= self.interval - self.tol:
            self.last = t
            return True
        return False


def even_size(width: float, height: float) -> Tuple[int, int]:
    """Round a render size down to even numbers (yuv420p needs even dimensions)."""
    return max(2, int(width) // 2 * 2), max(2, int(height) // 2 * 2)


class Mp4Writer:
    """Constant-frame-rate MP4 writer that preserves the source's real timing.

    Each frame is repeated until the output clock reaches the next source timestamp, so a VFR
    source keeps its duration and playback speed. Uses an ffmpeg pipe (H.264, yuv420p, faststart:
    plays in browsers); without ffmpeg falls back to OpenCV ``avc1`` and finally ``mp4v``
    (``browser_playable`` is then False).
    """

    def __init__(self, path: str, width: int, height: int, fps: float, ffmpeg: Optional[str] = None,
                 crf: int = 23):
        if width % 2 or height % 2:
            raise ValueError("width and height must be even")
        self.path, self.w, self.h, self.fps = str(path), width, height, float(fps)
        self.n_written = 0
        self._proc = None
        self._cv = None
        self._stderr = None
        self.codec: Optional[str] = None
        self.browser_playable = False
        exe = find_ffmpeg(ffmpeg)
        if exe:
            self._stderr = tempfile.TemporaryFile()
            self._proc = subprocess.Popen(
                [exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                 "-r", f"{self.fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr)
            self.codec, self.browser_playable = "h264", True
            return
        for fourcc, playable in (("avc1", True), ("mp4v", False)):
            cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*fourcc), self.fps, (width, height))
            if cv.isOpened():
                self._cv, self.codec, self.browser_playable = cv, fourcc, playable
                break
            cv.release()
        if self._cv is None:
            raise RuntimeError("no MP4 encoder available (install ffmpeg or imageio-ffmpeg)")
        if not self.browser_playable:
            logger.warning("ffmpeg not found: writing mp4v, which browsers cannot play")

    def _emit(self, frame: np.ndarray) -> None:
        if self._proc is not None:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self._cv.write(frame)
        self.n_written += 1

    def write(self, frame: np.ndarray, t_end: float) -> None:
        """Write ``frame`` until the output clock reaches ``t_end`` (the next frame's time, s)."""
        target = max(self.n_written + 1, int(round(t_end * self.fps)))
        while self.n_written < target:
            self._emit(frame)

    def _stderr_tail(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", "replace")[-800:]

    def close(self) -> None:
        if self._proc is not None:
            proc, self._proc = self._proc, None
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                code = proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
            tail = self._stderr_tail()
            self._stderr.close()
            if code != 0:
                raise RuntimeError(f"ffmpeg exited with {code}: {tail}")
        elif self._cv is not None:
            self._cv.release()
            self._cv = None

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
