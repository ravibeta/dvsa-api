"""Draw tracked vehicles, movement states, trails and a live counts panel on video frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .activity import MOVING, STATIONARY, UNKNOWN

# BGR. Separable from each other and from typical road / vegetation colours.
STATE_BGR = {MOVING: (0, 70, 255), STATIONARY: (255, 210, 0), UNKNOWN: (200, 200, 200)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
CLASS_SHORT = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "moto", "bicycle": "bike"}


@dataclass
class VehicleDraw:
    id: int
    class_name: str
    box: np.ndarray  # image xyxy (source pixels)
    state: str
    trail: Optional[np.ndarray] = None  # (N, 2) image points, oldest first


@dataclass
class PanelInfo:
    t: float
    duration: float
    frame: int
    visible: int
    detections: int
    unique_ids: int
    camera_ok: bool
    by_state: Dict[str, int] = field(default_factory=dict)
    by_class: Dict[str, int] = field(default_factory=dict)


def _fmt_t(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def draw_frame(frame: np.ndarray, items: List[VehicleDraw], panel: Optional[PanelInfo],
               out_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Return an annotated copy of ``frame`` resized to ``out_size`` ``(w, h)`` (default: same size)."""
    h0, w0 = frame.shape[:2]
    w, h = out_size or (w0, h0)
    out = frame.copy() if (w, h) == (w0, h0) else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    sx, sy = w / float(w0), h / float(h0)
    u = max(w, h) / 1280.0
    th = max(1, int(round(2 * u)))
    fs = 0.45 * u

    for it in items:
        if it.trail is not None and len(it.trail) >= 2:
            pts = np.round(np.asarray(it.trail) * [sx, sy]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, STATE_BGR.get(it.state, STATE_BGR[UNKNOWN]), th, cv2.LINE_AA)
    for it in items:
        col = STATE_BGR.get(it.state, STATE_BGR[UNKNOWN])
        x1, y1, x2, y2 = (int(round(v)) for v in (it.box[0] * sx, it.box[1] * sy, it.box[2] * sx, it.box[3] * sy))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)
        label = f"{CLASS_SHORT.get(it.class_name, it.class_name)} {it.id}"
        (tw, tht), _ = cv2.getTextSize(label, FONT, fs, 1)
        ty = y1 - 3 if y1 - tht - 4 >= 0 else y2 + tht + 3
        cv2.rectangle(out, (x1 - 1, ty - tht - 2), (x1 + tw + 1, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (x1, ty), FONT, fs, col, 1, cv2.LINE_AA)

    if panel is None:
        return out
    st = panel.by_state
    classes = "  ".join(f"{CLASS_SHORT.get(c, c)} {panel.by_class.get(c, 0)}" for c in CLASS_SHORT)
    lines = [
        (f"t {_fmt_t(panel.t)} / {_fmt_t(panel.duration)}   frame {panel.frame}", (255, 255, 255)),
        (f"vehicles tracked in view: {panel.visible}   (raw detections: {panel.detections})", (255, 255, 255)),
        (f"MOVING {st.get(MOVING, 0)}", STATE_BGR[MOVING]),
        (f"STATIONARY {st.get(STATIONARY, 0)}", STATE_BGR[STATIONARY]),
        (f"UNKNOWN {st.get(UNKNOWN, 0)}", STATE_BGR[UNKNOWN]),
        (classes, (255, 255, 255)),
        (f"unique IDs so far: {panel.unique_ids} (estimate, unvalidated)", (255, 255, 255)),
    ]
    if not panel.camera_ok:
        lines.append(("camera registration FAILED this frame", (0, 0, 255)))
    pfs = 0.5 * u
    lh = max(10, int(20 * u))
    pw = max(cv2.getTextSize(s, FONT, pfs, 1)[0][0] for s, _ in lines) + int(14 * u)
    ph = lh * len(lines) + int(10 * u)
    x0, y0 = int(6 * u), int(6 * u)
    roi = out[y0:min(h, y0 + ph), x0:min(w, x0 + pw)]
    roi[:] = (roi * 0.35).astype(out.dtype)
    for i, (s, col) in enumerate(lines):
        cv2.putText(out, s, (x0 + int(7 * u), y0 + int(5 * u) + lh * (i + 1) - int(4 * u)),
                    FONT, pfs, col, 1, cv2.LINE_AA)
    return out
