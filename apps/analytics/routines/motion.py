"""Motion analytics: background subtraction and optical flow.

Implements the classical motion techniques from ``plan-of-action.md`` §6.2.

* :func:`background_subtraction` uses an MOG2 model to extract moving
  foreground blobs across a sequence of frames (motion detection / foreground
  extraction).
* :func:`dense_optical_flow` uses Farneback flow between two frames to estimate
  scene dynamics (mean magnitude and dominant direction).

The background-subtraction routine is *video level*: it consumes an iterable of
frames rather than a single frame, since motion is only defined over time.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .base import Detection, register, to_gray


@register(
    "background_subtraction",
    "Detect moving foreground objects across frames (MOG2).",
    level="video",
)
def background_subtraction(
    frames: Iterable[np.ndarray],
    min_area: float = 200.0,
    history: int = 200,
    var_threshold: float = 16.0,
    detect_shadows: bool = True,
    **_: object,
) -> dict:
    """Detect moving objects across ``frames`` using MOG2.

    Returns per-frame moving-object detections plus an overall motion summary.
    """

    import cv2

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=detect_shadows
    )
    kernel = np.ones((5, 5), np.uint8)

    per_frame: List[dict] = []
    motion_ratios: List[float] = []

    for idx, frame in enumerate(frames):
        fg = subtractor.apply(frame)
        # Shadows are flagged as 127; keep only hard foreground.
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

        motion_ratios.append(float(np.count_nonzero(fg)) / float(fg.size))

        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        dets: List[Detection] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            dets.append(
                Detection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    centroid=(x + w / 2.0, y + h / 2.0),
                    area=area,
                    label="motion",
                )
            )
        per_frame.append({"frame": idx, "moving_objects": [d.to_dict() for d in dets]})

    return {
        "frames_processed": len(per_frame),
        "mean_motion_ratio": float(np.mean(motion_ratios)) if motion_ratios else 0.0,
        "max_motion_ratio": float(np.max(motion_ratios)) if motion_ratios else 0.0,
        "per_frame": per_frame,
    }


def dense_optical_flow(prev_frame: np.ndarray, next_frame: np.ndarray) -> dict:
    """Farneback dense optical flow between two frames.

    Returns mean/median magnitude and the dominant motion direction (degrees).
    """

    import cv2

    prev_gray = to_gray(prev_frame)
    next_gray = to_gray(next_frame)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    return {
        "mean_magnitude": float(mag.mean()),
        "median_magnitude": float(np.median(mag)),
        "max_magnitude": float(mag.max()),
        # Direction weighted by magnitude (dominant flow direction).
        "dominant_direction_deg": float(
            (ang * mag).sum() / mag.sum() if mag.sum() else 0.0
        ),
    }


@register(
    "optical_flow",
    "Dense Farneback optical flow between two consecutive frames.",
    level="frame_pair",
)
def optical_flow_routine(
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray] = None,
    **_: object,
) -> dict:
    if prev_frame is None:
        raise ValueError("optical_flow requires a 'prev_frame'")
    summary = dense_optical_flow(prev_frame, frame)
    return {"routine": "optical_flow", "summary": summary}
