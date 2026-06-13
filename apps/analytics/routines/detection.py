"""Object detection via classical color segmentation and contour analysis.

Implements the OpenCV pipeline called out in ``plan-of-action.md`` §1.1 and
§20 (bounding-box extraction): ``cvtColor`` -> ``inRange`` -> morphological
clean-up -> ``findContours`` -> ``moments`` for centroids. This provides
label-free, training-free detection of objects that are separable by color
(e.g. vehicles on tarmac, markers, vegetation) which is a common first-pass
analytic for aerial footage.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import Detection, RoutineResult, register, to_gray


def _contours_to_detections(
    contours: Sequence[np.ndarray], min_area: float, label: str
) -> List[Detection]:
    import cv2

    detections: List[Detection] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        m = cv2.moments(cnt)
        if m["m00"] != 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:  # degenerate contour — fall back to bbox centre
            cx, cy = x + w / 2.0, y + h / 2.0
        detections.append(
            Detection(
                bbox=(int(x), int(y), int(w), int(h)),
                centroid=(cx, cy),
                area=area,
                label=label,
            )
        )
    return detections


def detect_by_color(
    frame: np.ndarray,
    lower_hsv: Sequence[int],
    upper_hsv: Sequence[int],
    min_area: float = 100.0,
    kernel_size: int = 5,
    label: str = "object",
) -> List[Detection]:
    """Detect blobs whose HSV color falls within ``[lower_hsv, upper_hsv]``.

    Returns a list of :class:`Detection` sorted by descending area.
    """

    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8),
                       np.array(upper_hsv, dtype=np.uint8))

    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = _contours_to_detections(contours, min_area, label)
    dets.sort(key=lambda d: d.area, reverse=True)
    return dets


def detect_by_threshold(
    frame: np.ndarray,
    min_area: float = 100.0,
    block_size: int = 35,
    c: int = 5,
    invert: bool = True,
    label: str = "object",
) -> List[Detection]:
    """Detect foreground blobs via adaptive thresholding on a grayscale frame.

    Useful when objects are separable by intensity rather than hue. Uses an
    adaptive Gaussian threshold so it is robust to uneven aerial lighting.
    """

    import cv2

    gray = to_gray(frame)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    # block_size must be odd and > 1.
    bs = block_size if block_size % 2 == 1 else block_size + 1
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, bs, c
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = _contours_to_detections(contours, min_area, label)
    dets.sort(key=lambda d: d.area, reverse=True)
    return dets


@register(
    "color_detection",
    "Detect & count objects by HSV color segmentation (bbox + centroid).",
    level="frame",
)
def color_detection_routine(
    frame: np.ndarray,
    lower_hsv: Optional[Sequence[int]] = None,
    upper_hsv: Optional[Sequence[int]] = None,
    min_area: float = 100.0,
    label: str = "object",
    **_: object,
) -> dict:
    """Routine wrapper around :func:`detect_by_color`.

    Defaults select reddish hues (wrapping is not handled; tune per dataset).
    """

    if lower_hsv is None:
        lower_hsv = [0, 70, 50]
    if upper_hsv is None:
        upper_hsv = [10, 255, 255]

    dets = detect_by_color(frame, lower_hsv, upper_hsv, min_area=min_area, label=label)
    result = RoutineResult(
        routine="color_detection",
        summary={"count": len(dets), "label": label},
        detections=dets,
    )
    return result.to_dict()


@register(
    "threshold_detection",
    "Detect & count objects by adaptive intensity thresholding.",
    level="frame",
)
def threshold_detection_routine(
    frame: np.ndarray,
    min_area: float = 100.0,
    invert: bool = True,
    label: str = "object",
    **_: object,
) -> dict:
    dets = detect_by_threshold(frame, min_area=min_area, invert=invert, label=label)
    result = RoutineResult(
        routine="threshold_detection",
        summary={"count": len(dets), "label": label},
        detections=dets,
    )
    return result.to_dict()
