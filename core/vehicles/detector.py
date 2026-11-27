"""Vehicle detectors.

:class:`VehicleDetector` is the interface the pipeline uses (tests inject a fake). The default
implementation wraps a COCO-pretrained Ultralytics YOLO model (AGPL-3.0, accepted for dvsa-api) and
keeps only vehicle classes. ``ultralytics``/``torch`` are imported lazily; when they are missing or
the weights cannot be loaded, :class:`VehiclesUnavailable` is raised (the endpoint maps it to 503).
"""

from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

from .boxes import nms, tile_windows, touches_interior_edge

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")


class VehiclesUnavailable(RuntimeError):
    """The detector runtime (ultralytics/torch or the weights) is not available on this host."""


class VehicleDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return ``(boxes_xyxy (N,4) float64, scores (N,) float64, class_names list[str])``."""

    def describe(self) -> dict:
        """Model description for the report."""


def runtime_missing() -> Optional[str]:
    """Name the missing detector package, or ``None`` when ultralytics and torch are importable."""
    for name in ("torch", "ultralytics"):
        if importlib.util.find_spec(name) is None:
            return name
    return None


@dataclass
class DetectorConfig:
    # yolo26x at 1920 px: on high-altitude nadir city footage (vehicles ~15-25 px) yolo26m found
    # 0-3 vehicles per frame where yolo26x found ~20 (see docs/vehicle_tracking_plan.md).
    weights: str = "yolo26x.pt"
    model_dir: Optional[str] = None  # where weights are stored/downloaded; None = ultralytics default
    device: Optional[str] = None  # None/"" = auto (cuda:0 > mps > cpu)
    # Network input size. 0 = auto: ``upscale`` x the frame's long side (multiple of 32, clamped to
    # [640, 3840]). Small top-down vehicles need upsampling: 1.5x gave the most confident detections
    # on both 720p (-> 1920) and 1080p (-> 2880) nadir test footage.
    imgsz: int = 0
    upscale: float = 1.5
    conf: float = 0.10  # low on purpose: the tracker's second stage uses 0.10-0.25 detections
    classes: Tuple[str, ...] = VEHICLE_CLASSES
    max_det: int = 1000
    # Drop boxes whose long side exceeds this fraction of the frame's long side (whole building
    # blocks / rooftops reported as "truck" in nadir footage). 0 = off.
    max_box_frac: float = 0.25
    half: bool = True  # fp16 on CUDA only
    tile: bool = False
    tile_size: int = 1280
    tile_overlap: float = 0.2
    merge_iou: float = 0.5


def auto_imgsz(frame_shape: Optional[Tuple[int, ...]], upscale: float = 1.5) -> int:
    """Input size ``upscale`` x the frame's long side, a multiple of 32 in [640, 3840]."""
    if not frame_shape:
        return 1920
    size = int(round(max(frame_shape[:2]) * upscale / 32.0)) * 32
    return int(min(3840, max(640, size)))


def select_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class UltralyticsVehicleDetector:
    def __init__(self, config: Optional[DetectorConfig] = None, warmup_shape: Optional[Tuple[int, ...]] = None):
        self.config = cfg = config or DetectorConfig()
        missing = runtime_missing()
        if missing:
            raise VehiclesUnavailable(f"{missing} is not installed (pip install -r requirements/vehicles.txt)")
        self.timings_ms = {}
        t0 = time.perf_counter()
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            self.device = select_device(cfg.device)
            weights = cfg.weights
            if cfg.model_dir and not os.path.isabs(weights) and os.path.dirname(weights) == "":
                os.makedirs(cfg.model_dir, exist_ok=True)
                weights = os.path.join(cfg.model_dir, weights)
            self.model = YOLO(weights)  # downloads official weights on first use
            self.model.to(self.device)
        except VehiclesUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - bad weights name, no network, CUDA init failure, ...
            raise VehiclesUnavailable(f"cannot load detector weights {cfg.weights!r}: {exc}") from exc
        self.timings_ms["model_load"] = (time.perf_counter() - t0) * 1e3
        names = self.model.names
        wanted = {c.lower() for c in cfg.classes}
        self.class_ids = sorted(i for i, n in names.items() if str(n).lower() in wanted)
        if not self.class_ids:
            raise VehiclesUnavailable(f"model {cfg.weights!r} has none of the classes {cfg.classes}")
        self.names = {i: str(names[i]) for i in self.class_ids}
        self.fp16 = bool(cfg.half and self.device.startswith("cuda"))
        self.imgsz = cfg.imgsz if cfg.imgsz and cfg.imgsz > 0 else auto_imgsz(warmup_shape, cfg.upscale)
        t0 = time.perf_counter()
        self.detect(np.zeros(warmup_shape or (self.imgsz, self.imgsz, 3), dtype=np.uint8))
        self.timings_ms["warmup"] = (time.perf_counter() - t0) * 1e3

    def _predict(self, images: List[np.ndarray]):
        cfg = self.config
        kwargs = dict(imgsz=self.imgsz, conf=cfg.conf, classes=self.class_ids, agnostic_nms=True,
                      max_det=cfg.max_det, device=self.device, verbose=False)
        if self.fp16:
            kwargs["quantize"] = 16
        return self.model.predict(images, **kwargs)

    def _arrays(self, result) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        b = result.boxes
        boxes = b.xyxy.cpu().numpy().astype(np.float64).reshape(-1, 4)
        scores = b.conf.cpu().numpy().astype(np.float64).reshape(-1)
        cls = b.cls.cpu().numpy().astype(int).reshape(-1)
        return boxes, scores, [self.names.get(int(c), "vehicle") for c in cls]

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        cfg = self.config
        boxes, scores, names = self._arrays(self._predict([frame_bgr])[0])
        h, w = frame_bgr.shape[:2]
        if cfg.tile and max(h, w) > cfg.tile_size:
            windows = tile_windows(w, h, cfg.tile_size, cfg.tile_overlap)
            all_b, all_s, all_n = [boxes], [scores], list(names)
            for win, res in zip(windows, self._predict([frame_bgr[y0:y1, x0:x1] for x0, y0, x1, y1 in windows])):
                tb, ts, tn = self._arrays(res)
                tb[:, [0, 2]] += win[0]
                tb[:, [1, 3]] += win[1]
                keep = ~touches_interior_edge(tb, win, w, h)
                all_b.append(tb[keep])
                all_s.append(ts[keep])
                all_n.extend(n for n, k in zip(tn, keep) if k)
            boxes, scores = np.concatenate(all_b), np.concatenate(all_s)
            idx = nms(boxes, scores, cfg.merge_iou)[: cfg.max_det]
            boxes, scores, names = boxes[idx], scores[idx], [all_n[i] for i in idx]
        if cfg.max_box_frac and len(boxes):
            long_side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
            keep = long_side <= cfg.max_box_frac * max(h, w)
            boxes, scores, names = boxes[keep], scores[keep], [n for n, k in zip(names, keep) if k]
        return boxes, scores, names

    def describe(self) -> dict:
        d = asdict(self.config)
        d.update({"type": "ultralytics-coco", "imgsz": self.imgsz, "device": self.device, "fp16": self.fp16,
                  "classes": list(self.names.values()),
                  "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()}})
        return d
