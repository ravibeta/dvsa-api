"""Synthetic drone footage for the vehicle tracking tests (no model weights, no network).

``make_scene_video`` renders a textured "ground" seen by a panning camera, with solid red
rectangles as vehicles (some parked, some driving). :class:`ColourDetector` finds those rectangles by
colour, so the whole pipeline -- camera compensation, tracking, activity, rendering, encoding -- runs
for real while the detector stays deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class SynthVehicle:
    x: float  # ground px (centre) at t=0
    y: float
    vx: float = 0.0  # ground px / s
    vy: float = 0.0
    w: float = 40.0
    h: float = 20.0


def texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 2.0)
    return cv2.cvtColor(cv2.equalizeHist(img), cv2.COLOR_GRAY2BGR)


def make_scene_video(path: str, vehicles: Sequence[SynthVehicle], seconds: float = 4.0, fps: float = 30.0,
                     size: Tuple[int, int] = (640, 360), pan_px_s: Tuple[float, float] = (40.0, 10.0),
                     seed: int = 0) -> List[List[Tuple[float, float, float, float]]]:
    """Write an MP4 (OpenCV ``mp4v``) and return per-frame ground-truth image boxes (xyxy)."""
    w, h = size
    n = int(round(seconds * fps))
    ground = texture(h + int(abs(pan_px_s[1]) * seconds) + 40, w + int(abs(pan_px_s[0]) * seconds) + 40, seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV cannot write mp4v here"
    gt = []
    for k in range(n):
        t = k / fps
        cx, cy = 20 + pan_px_s[0] * t, 20 + pan_px_s[1] * t  # camera origin in ground px
        M = np.float32([[1, 0, -cx], [0, 1, -cy]])
        frame = cv2.warpAffine(ground, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        boxes = []
        for v in vehicles:
            gx, gy = v.x + v.vx * t, v.y + v.vy * t
            x1, y1 = gx - v.w / 2 - cx, gy - v.h / 2 - cy
            x2, y2 = x1 + v.w, y1 + v.h
            cv2.rectangle(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                          (0, 0, 230), -1)
            boxes.append((x1, y1, x2, y2))
        writer.write(frame)
        gt.append(boxes)
    writer.release()
    return gt


class ColourDetector:
    """Detects the synthetic red vehicles; implements ``core.vehicles.detector.VehicleDetector``."""

    def __init__(self, class_name: str = "car", score: float = 0.9, min_area: int = 80):
        self.class_name, self.score, self.min_area = class_name, score, min_area
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray):
        self.calls += 1
        b, g, r = cv2.split(frame_bgr.astype(np.int16))
        mask = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = [[x, y, x + bw, y + bh] for x, y, bw, bh, area in stats[1:] if area >= self.min_area]
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        return boxes, np.full(len(boxes), self.score), [self.class_name] * len(boxes)

    def describe(self) -> dict:
        return {"type": "synthetic-colour"}
