"""Extract the four survey-area corner frames from a drone video.

Ported from the workspace reference script ``survey_corners.py`` into a reusable,
side-effect-free runner so it can back the ``/corners`` endpoint directly —
bypassing the query / RAG / agentic path. Two modes, unchanged from the script:

  telemetry -- if a DJI-style ``.SRT`` sidecar with ``[latitude]``/``[longitude]``
               is present, the flight track comes straight from GPS.
  visual    -- otherwise, estimate the track by accumulating frame-to-frame
               similarity transforms (rough visual odometry on the ground plane).

The track is reduced to its minimum-area rotated rectangle and the frame nearest
each rectangle corner is emitted, ordered bottom-left then clockwise
(BL -> TL -> TR -> BR).

``cv2`` is imported lazily inside the functions that need it, so importing this
module stays cheap and the pure geometry helpers are unit-testable without a
video runtime.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("apps.azure")

SRT_LAT = re.compile(r"\[latitude\s*:\s*([-\d.]+)\]")
SRT_LON = re.compile(r"\[long(?:i)?tude\s*:\s*([-\d.]+)\]")

# Emitted order: bottom-left, top-left, top-right, bottom-right.
CORNER_LABELS = ["1-bottom-left", "2-top-left", "3-top-right", "4-bottom-right"]


# ---------------------------------------------------------------- telemetry
def track_from_srt(path: str):
    """Return an ``Nx2`` array of local metres (east, north), or ``None``."""
    text = open(path, "r", errors="ignore").read()
    lats = [float(m) for m in SRT_LAT.findall(text)]
    lons = [float(m) for m in SRT_LON.findall(text)]
    if len(lats) < 4 or len(lats) != len(lons):
        return None
    lat0 = math.radians(np.mean(lats))
    # equirectangular projection, fine over a survey-sized area
    east = (np.array(lons) - np.mean(lons)) * 111320.0 * math.cos(lat0)
    north = (np.array(lats) - np.mean(lats)) * 110540.0
    return np.column_stack([east, north])


# ------------------------------------------------------------------ visual
def track_from_video(cap, sample_fps: float, width: int):
    """Accumulate similarity transforms into a rough ground track.

    World frame is aligned to the first frame's heading, so 'bottom-left' is
    relative to the drone's initial orientation, not to true north.
    """
    import cv2  # noqa: PLC0415

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / sample_fps)))

    orb = cv2.ORB_create(1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    positions, frame_idx = [], []
    pos = np.zeros(2)
    heading = 0.0  # cumulative yaw, radians
    prev_kp = prev_des = None
    scale = None
    i = 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        if scale is None:
            scale = width / float(frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        kp, des = orb.detectAndCompute(gray, None)

        if prev_des is not None and des is not None and len(des) > 10:
            matches = matcher.match(prev_des, des)
            if len(matches) >= 12:
                matches = sorted(matches, key=lambda m: m.distance)[:400]
                src = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                dst = np.float32([kp[m.trainIdx].pt for m in matches])
                M, _ = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                if M is not None:
                    # scene shift in image space; camera moves the other way
                    dx, dy = -M[0, 2], -M[1, 2]
                    dyaw = math.atan2(M[1, 0], M[0, 0])
                    c, s = math.cos(heading), math.sin(heading)
                    # rotate into world frame, flip y so +y is "up" on the map
                    pos = pos + np.array([c * dx - s * dy, -(s * dx + c * dy)])
                    heading += dyaw

        positions.append(pos.copy())
        frame_idx.append(i)
        prev_kp, prev_des = kp, des
        i += 1

    return np.array(positions), np.array(frame_idx)


# ------------------------------------------------------------------ corners
def order_clockwise_from_bottom_left(box, centre):
    """Order 4 points BL -> TL -> TR -> BR in a y-up coordinate frame."""
    ang = np.array([math.atan2(p[1] - centre[1], p[0] - centre[0]) % (2 * math.pi)
                    for p in box])
    target = 5 * math.pi / 4  # 225 deg = bottom-left
    diff = np.abs((ang - target + math.pi) % (2 * math.pi) - math.pi)
    start = int(np.argmin(diff))
    order = list(np.argsort(-ang))          # clockwise = decreasing angle
    k = order.index(start)
    return [box[j] for j in order[k:] + order[:k]]


def corner_samples(track):
    """Indices into ``track`` of the four survey-area corners, BL-first clockwise."""
    import cv2  # noqa: PLC0415

    pts = track.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    ordered = order_clockwise_from_bottom_left(box, np.array(rect[0]))
    return [int(np.argmin(np.linalg.norm(pts - c, axis=1))) for c in ordered]


def grab_frame(cap, index):
    import cv2  # noqa: PLC0415

    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


# --------------------------------------------------------------------- entry
def extract_corner_frames(
    video_path: str,
    srt_path: Optional[str] = None,
    fps: float = 2.0,
    width: int = 640,
) -> List[Tuple[str, bytes, Dict[str, Any]]]:
    """Return the four corner frames as ``(label, jpeg_bytes, meta)`` tuples.

    ``meta`` carries ``frame`` (index), ``t`` (seconds), ``x``/``y`` (track
    metres) and ``mode`` (``telemetry`` or ``visual``). Raises ``RuntimeError``
    if the video can't be opened or a track can't be estimated. A frame that
    can't be read or encoded is skipped rather than aborting the whole request.
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        srt = srt_path or (os.path.splitext(video_path)[0] + ".srt")
        track = track_from_srt(srt) if srt and os.path.exists(srt) else None

        if track is not None:
            mode = "telemetry"
            frame_idx = np.linspace(0, max(total - 1, 0), len(track)).astype(int)
        else:
            mode = "visual"
            track, frame_idx = track_from_video(cap, fps, width)
            if len(track) < 8:
                raise RuntimeError("not enough usable frames to estimate a track")

        picks = corner_samples(track)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        results: List[Tuple[str, bytes, Dict[str, Any]]] = []
        for label, s in zip(CORNER_LABELS, picks):
            fi = int(frame_idx[s])
            frame = grab_frame(cap, fi)
            if frame is None:
                logger.info("corners: frame %s (%s) unreadable", fi, label)
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                logger.info("corners: frame %s (%s) failed to encode", fi, label)
                continue
            results.append((label, buf.tobytes(), {
                "frame": fi,
                "t": round(fi / src_fps, 2),
                "x": float(track[s][0]),
                "y": float(track[s][1]),
                "mode": mode,
            }))
        return results
    finally:
        cap.release()
