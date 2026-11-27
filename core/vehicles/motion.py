"""Camera (drone) motion estimation and compensation.

Each processed frame is registered to the previous one with background features: Shi-Tomasi
corners -> pyramidal Lucas-Kanade flow -> RANSAC fit of a global 2-D transform. Corners on or near
detected vehicles are masked out, because vehicles move independently of the road. Chaining the
frame-to-frame transforms gives ``H_cur_to_ref``, which maps the current image into the image plane
of the first processed frame ("ref" / ground coordinates). A parked vehicle keeps the same ref
position however the drone pans, rotates or changes altitude.

Assumes an approximately planar ground. ``similarity`` suits near-nadir footage; ``homography`` adds
perspective for oblique footage. Chained registration drifts slowly, so speeds are measured over
short windows. Too few background features or a poor fit marks the step ``ok=False`` and reuses the
last good motion (constant camera velocity); states that depend on it become UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

MOTION_MODELS = ("similarity", "affine", "homography", "none")


@dataclass
class MotionStep:
    H_prev_to_cur: np.ndarray
    H_cur_to_ref: np.ndarray
    ok: bool
    n_features: int
    n_inliers: int
    residual_px: float

    def to_dict(self) -> dict:
        return {"ok": self.ok, "features": self.n_features, "inliers": self.n_inliers,
                "residual_px": None if np.isnan(self.residual_px) else round(float(self.residual_px), 3)}


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 transform to (N, 2) points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    return p[:, :2] / p[:, 2:3]


def transform_boxes(H: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding boxes of the four transformed corners of each xyxy box."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes.copy()
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                        np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)
    t = apply_h(H, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.concatenate([t.min(axis=1), t.max(axis=1)], axis=1)


class CameraMotionEstimator:
    def __init__(self, model: str = "similarity", max_features: int = 1500, ransac_px: float = 1.0,
                 min_inliers: int = 40, max_width: int = 1280, border_px: int = 8, mask_pad: float = 0.5):
        if model not in MOTION_MODELS:
            raise ValueError(f"motion model must be one of {MOTION_MODELS}")
        self.model, self.max_features, self.ransac_px = model, max_features, ransac_px
        self.min_inliers, self.max_width = min_inliers, max_width
        self.border_px, self.mask_pad = border_px, mask_pad
        self.H_cur_to_ref = np.eye(3)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_mask: Optional[np.ndarray] = None
        self._scale = 1.0
        self._last_good = np.eye(3)  # last successful frame-to-frame motion (full-res pixels)

    def _prepare(self, frame_bgr: np.ndarray, exclude_boxes) -> Tuple[np.ndarray, np.ndarray]:
        w = frame_bgr.shape[1]
        self._scale = min(1.0, self.max_width / float(w))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._scale < 1.0:
            gray = cv2.resize(gray, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        gh, gw = gray.shape
        mask = np.zeros((gh, gw), np.uint8)
        b = self.border_px
        mask[b:gh - b, b:gw - b] = 255  # borders often carry overlays / compression artefacts
        if exclude_boxes is not None and len(exclude_boxes):
            for x1, y1, x2, y2 in np.asarray(exclude_boxes, dtype=np.float64).reshape(-1, 4) * self._scale:
                pw, ph = (x2 - x1) * self.mask_pad, (y2 - y1) * self.mask_pad
                cv2.rectangle(mask, (int(x1 - pw), int(y1 - ph)), (int(x2 + pw), int(y2 + ph)), 0, -1)
        return gray, mask

    def update(self, frame_bgr: np.ndarray, exclude_boxes=None) -> MotionStep:
        """Register ``frame_bgr`` to the previous frame given to this method.

        ``exclude_boxes`` are this frame's detections (full-resolution xyxy); they mask features when
        this frame becomes the *previous* frame of the next update.
        """
        gray, mask = self._prepare(frame_bgr, exclude_boxes)
        prev, prev_mask = self._prev_gray, self._prev_mask
        self._prev_gray, self._prev_mask = gray, mask
        if prev is None or self.model == "none":
            return MotionStep(np.eye(3), self.H_cur_to_ref.copy(), True, 0, 0, float("nan"))

        A, n_inl, resid = None, 0, float("nan")
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=self.max_features, qualityLevel=0.005, minDistance=8,
                                     mask=prev_mask, blockSize=7)
        n_feat = 0 if p0 is None else len(p0)
        if n_feat >= self.min_inliers:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                prev, gray, p0, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            good = st.ravel() == 1
            a, b = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
            if len(a) >= self.min_inliers:
                if self.model == "homography":
                    M, inl = cv2.findHomography(a, b, cv2.RANSAC, self.ransac_px, maxIters=2000, confidence=0.995)
                else:
                    fit = cv2.estimateAffinePartial2D if self.model == "similarity" else cv2.estimateAffine2D
                    M, inl = fit(a, b, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_px,
                                 maxIters=2000, confidence=0.995)
                    M = None if M is None else np.vstack([M, [0.0, 0.0, 1.0]])
                if M is not None and inl is not None:
                    inl = inl.ravel().astype(bool)
                    n_inl = int(inl.sum())
                    if n_inl:
                        resid = float(np.median(np.linalg.norm(apply_h(M, a[inl]) - b[inl], axis=1)))
                    A = M
        ok = A is not None and n_inl >= self.min_inliers
        if ok:
            S = np.diag([self._scale, self._scale, 1.0])
            A = np.linalg.inv(S) @ A @ S  # back to full-resolution pixels
            resid = resid / self._scale
            self._last_good = A
        else:
            # Unknown motion (blur, glitch frame): assume the camera kept its last motion, so one bad
            # step does not shift every track's ground position; the step is still flagged.
            A, resid = self._last_good.copy(), float("nan")
        self.H_cur_to_ref = self.H_cur_to_ref @ np.linalg.inv(A)
        self.H_cur_to_ref /= self.H_cur_to_ref[2, 2]
        return MotionStep(A, self.H_cur_to_ref.copy(), ok, n_feat, n_inl, resid)
