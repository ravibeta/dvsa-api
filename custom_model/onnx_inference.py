"""Custom ONNX object-detection adapter.

Implements :class:`CustomONNXDetector`, a small, dependency-injectable wrapper
around an ``onnxruntime.InferenceSession`` that turns a BGR frame (as returned
by OpenCV) into a list of detection dicts::

    {"label": "vehicle", "score": 0.92, "bbox": [x, y, w, h]}

``bbox`` is ``(x, y, w, h)`` — top-left corner plus width/height — in
**original frame pixel space**, matching ``apps.analytics.routines.base.Detection``
used elsewhere in dvsa-api. This holds regardless of the model's input size or
whether tiling is used. Models almost always emit corner boxes
(``[x1, y1, x2, y2]``); the adapter converts on the way out (see
:func:`xyxy_to_xywh`).

The design deliberately allows the runtime session to be *injected* (either as
a ready ``session`` or via a ``session_factory``) so the full pipeline can be
unit-tested without a real ``.onnx`` binary — see ``tests/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .model_loader import ModelConfig, load_label_map, validate_model_file

logger = logging.getLogger(__name__)

# Heuristic: if every box coordinate is <= this value we treat the boxes as
# normalised to [0, 1] rather than expressed in input-pixel space.
_NORMALISED_COORD_MAX = 1.5

PreprocessOutput = Union[np.ndarray, Tuple[List[np.ndarray], List["TileMeta"]]]


@dataclass
class TileMeta:
    """Maps a model input back onto a region of the original frame.

    ``(x_off, y_off)`` is the top-left of the region in original-frame pixels
    and ``(width, height)`` its size, *before* the region was resized to the
    model input size.
    """

    x_off: int
    y_off: int
    width: int
    height: int


class CustomONNXDetector:
    """Run a custom ONNX object-detection model over BGR frames.

    Parameters
    ----------
    config:
        The :class:`~custom_model.model_loader.ModelConfig`.
    session:
        An already-constructed ONNX session (or any object exposing the
        ``run``/``get_inputs``/``get_outputs`` API). Mainly for tests.
    session_factory:
        Callable ``(onnx_path) -> session`` used by :meth:`load` to build the
        session lazily. Takes precedence over creating a real
        ``onnxruntime.InferenceSession`` only when provided.
    """

    def __init__(
        self,
        config: ModelConfig,
        session: Optional[object] = None,
        session_factory: Optional[Callable[[str], object]] = None,
    ) -> None:
        self.config = config
        self._session = session
        self._session_factory = session_factory
        self._label_map: Optional[Dict[int, str]] = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[List[str]] = None
        # Original (height, width) of the most recently preprocessed frame.
        self._orig_shape: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> "CustomONNXDetector":
        """Build the ONNX session (if needed) and load the label map.

        Returns ``self`` so calls can be chained: ``create_detector(cfg).load()``.
        """

        self._label_map = load_label_map(self.config.labels_path)

        if self._session is None:
            if self._session_factory is not None:
                logger.info("Creating ONNX session via injected session_factory")
                self._session = self._session_factory(self.config.onnx_path)
            else:
                validate_model_file(self.config.onnx_path)
                logger.info("Creating onnxruntime.InferenceSession for %s", self.config.onnx_path)
                import onnxruntime  # imported lazily; not needed when a session is injected

                self._session = onnxruntime.InferenceSession(
                    self.config.onnx_path,
                    providers=["CPUExecutionProvider"],
                )

        # Cache input/output names for feeding the session and for diagnostics.
        try:
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
        except Exception:  # pragma: no cover - defensive; mocks may differ
            self._input_name = self._input_name or "input"
            self._output_names = self._output_names or None

        logger.info(
            "Detector loaded: %d labels, input=%s, outputs=%s",
            len(self._label_map or {}),
            self._input_name,
            self._output_names,
        )
        return self

    def close(self) -> None:
        """Release the ONNX session. Safe to call multiple times."""

        # onnxruntime sessions hold native resources but expose no explicit
        # close(); dropping the reference lets them be garbage collected.
        self._session = None
        logger.info("Detector closed")

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def _to_tensor(self, region: np.ndarray) -> np.ndarray:
        """Resize ``region`` to the model input and return a ``(1, C, H, W)`` tensor."""

        import cv2

        in_w, in_h = self.config.input_size
        resized = cv2.resize(region, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        # BGR (OpenCV) -> RGB.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Scale to [0, 1] then normalise per channel.
        arr = rgb.astype(np.float32) / 255.0
        mean = np.asarray(self.config.mean, dtype=np.float32)
        std = np.asarray(self.config.std, dtype=np.float32)
        arr = (arr - mean) / std
        # HWC -> CHW and add batch dimension.
        chw = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def preprocess(self, frame: np.ndarray) -> PreprocessOutput:
        """Convert a BGR frame into model input tensor(s).

        Without tiling, returns a single ``(1, C, H, W)`` ``float32`` tensor.

        With ``config.tile_size`` set, returns ``(tensors, tile_metadata)``
        where ``tensors`` is a list of ``(1, C, H, W)`` tensors (one per tile)
        and ``tile_metadata`` describes how to map each tile back to the
        original frame.
        """

        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be an HxWx3 BGR numpy array")

        h, w = frame.shape[:2]
        self._orig_shape = (h, w)

        if self.config.tile_size is None:
            logger.debug("Preprocess (no tiling): frame %dx%d -> %s", w, h, self.config.input_size)
            return self._to_tensor(frame)

        tensors: List[np.ndarray] = []
        metas: List[TileMeta] = []
        for region, meta in self._generate_tiles(frame):
            tensors.append(self._to_tensor(region))
            metas.append(meta)
        logger.debug("Preprocess (tiling): frame %dx%d -> %d tiles", w, h, len(tensors))
        return tensors, metas

    def _generate_tiles(self, frame: np.ndarray):
        """Yield ``(region, TileMeta)`` covering ``frame`` with overlap."""

        h, w = frame.shape[:2]
        tile_w, tile_h = self.config.tile_size  # type: ignore[misc]
        overlap = self.config.tile_overlap
        step_x = max(1, int(tile_w * (1.0 - overlap)))
        step_y = max(1, int(tile_h * (1.0 - overlap)))

        ys = list(range(0, max(1, h - 1), step_y))
        xs = list(range(0, max(1, w - 1), step_x))
        for y0 in ys:
            for x0 in xs:
                x1 = min(x0 + tile_w, w)
                y1 = min(y0 + tile_h, h)
                region = frame[y0:y1, x0:x1]
                if region.size == 0:
                    continue
                yield region, TileMeta(x_off=x0, y_off=y0, width=x1 - x0, height=y1 - y0)

    # ------------------------------------------------------------------ #
    # Postprocessing
    # ------------------------------------------------------------------ #

    def _parse_raw_outputs(
        self, raw_outputs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalise assorted ONNX output shapes into ``(boxes, scores, labels)``.

        Supported layouts
        ------------------
        * ``[boxes(N,4), scores(N,), labels(N,)]`` — three separate arrays.
        * A single ``(N, 6)`` / ``(1, N, 6)`` array of ``[x1,y1,x2,y2,score,cls]``.
        * A ``dict`` keyed by ``boxes``/``scores``/``labels`` (or the common
          TF-style ``detection_boxes``/``detection_scores``/``detection_classes``).

        Raises
        ------
        RuntimeError
            If the outputs do not match any supported layout. The message
            includes the model's output names and the observed shapes to aid
            debugging.
        """

        def _shapes(obj) -> str:
            try:
                if isinstance(obj, dict):
                    return {k: np.asarray(v).shape for k, v in obj.items()}  # type: ignore[return-value]
                return [np.asarray(o).shape for o in obj]
            except Exception:
                return repr(type(obj))

        names = self._output_names

        # dict output -------------------------------------------------------
        if isinstance(raw_outputs, dict):
            boxes = _first_present(raw_outputs, ("boxes", "detection_boxes", "dets"))
            scores = _first_present(raw_outputs, ("scores", "detection_scores"))
            labels = _first_present(raw_outputs, ("labels", "classes", "detection_classes"))
            if boxes is None or scores is None or labels is None:
                raise RuntimeError(
                    "Unexpected ONNX dict output. Expected keys for boxes/scores/labels; "
                    f"got keys={list(raw_outputs)} names={names} shapes={_shapes(raw_outputs)}"
                )
            return np.asarray(boxes), np.asarray(scores).reshape(-1), np.asarray(labels).reshape(-1)

        # Wrap a bare array so the sequence handling below applies uniformly.
        if isinstance(raw_outputs, np.ndarray):
            raw_outputs = [raw_outputs]

        if not isinstance(raw_outputs, Sequence):
            raise RuntimeError(
                f"Unexpected ONNX output type {type(raw_outputs).__name__}; "
                f"expected list/tuple/dict/ndarray. names={names}"
            )

        outs = [np.asarray(o) for o in raw_outputs]

        # Three-array layout ----------------------------------------------
        if len(outs) >= 3:
            boxes, scores, labels = outs[0], outs[1], outs[2]
            boxes = boxes.reshape(-1, boxes.shape[-1]) if boxes.size else boxes.reshape(0, 4)
            if boxes.shape[-1] != 4:
                raise RuntimeError(
                    "Unexpected ONNX boxes shape; expected (..., 4). "
                    f"names={names} shapes={_shapes(outs)}"
                )
            return boxes, scores.reshape(-1), labels.reshape(-1)

        # Single combined array [x1,y1,x2,y2,score,cls] -------------------
        if len(outs) == 1:
            arr = outs[0]
            arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(1, -1)
            if arr.shape[-1] >= 6:
                boxes = arr[:, 0:4]
                scores = arr[:, 4]
                labels = arr[:, 5]
                return boxes, scores.reshape(-1), labels.reshape(-1)

        raise RuntimeError(
            "Unsupported ONNX output layout; could not derive boxes/scores/labels. "
            f"names={names} shapes={_shapes(outs)}"
        )

    def _scale_box(self, box: Sequence[float], meta: TileMeta) -> List[int]:
        """Map a raw model corner-box onto original-frame ``(x, y, w, h)`` pixels."""

        bx1, by1, bx2, by2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        in_w, in_h = self.config.input_size

        if max(abs(bx1), abs(by1), abs(bx2), abs(by2)) <= _NORMALISED_COORD_MAX:
            # Boxes normalised to [0, 1] of the model input / region.
            sx, sy = meta.width, meta.height
        else:
            # Boxes in input-pixel space; rescale to the region size.
            sx, sy = meta.width / in_w, meta.height / in_h

        x1 = meta.x_off + bx1 * sx
        y1 = meta.y_off + by1 * sy
        x2 = meta.x_off + bx2 * sx
        y2 = meta.y_off + by2 * sy
        # Order corners, clamp to the frame origin, then convert to (x, y, w, h)
        # to match apps.analytics.routines.base.Detection.
        x1, x2 = sorted((max(0.0, x1), max(0.0, x2)))
        y1, y2 = sorted((max(0.0, y1), max(0.0, y2)))
        return xyxy_to_xywh((x1, y1, x2, y2))

    def postprocess(
        self,
        raw_outputs,
        tile_metadata: Optional[TileMeta] = None,
    ) -> List[Dict]:
        """Convert one session output into a list of detection dicts.

        Parameters
        ----------
        raw_outputs:
            The object returned by ``session.run`` for a single input tensor.
        tile_metadata:
            The :class:`TileMeta` describing the region of the original frame
            this output corresponds to. When ``None``, the whole most-recently
            preprocessed frame is assumed.
        """

        if self._label_map is None:
            raise RuntimeError("Detector not loaded; call load() before postprocess()")

        if tile_metadata is None:
            if self._orig_shape is None:
                raise RuntimeError("No frame shape available; run preprocess()/infer() first")
            h, w = self._orig_shape
            tile_metadata = TileMeta(x_off=0, y_off=0, width=w, height=h)

        boxes, scores, labels = self._parse_raw_outputs(raw_outputs)

        detections: List[Dict] = []
        for box, score, label_id in zip(boxes, scores, labels):
            score_f = float(score)
            if score_f < self.config.score_threshold:
                continue
            cls = int(round(float(label_id)))
            if cls not in self._label_map:
                raise ValueError(
                    f"Model produced class id {cls} which is absent from the label map "
                    f"(known ids: {sorted(self._label_map)}). Check label_map.json matches the model."
                )
            detections.append(
                {
                    "label": self._label_map[cls],
                    "score": score_f,
                    "bbox": self._scale_box(box, tile_metadata),
                }
            )
        return detections

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _run_session(self, tensor: np.ndarray):
        feed = {self._input_name or "input": tensor}
        return self._session.run(self._output_names, feed)  # type: ignore[union-attr]

    def infer(self, frame: np.ndarray) -> List[Dict]:
        """Run the full pipeline on ``frame`` and return detection dicts.

        Steps: :meth:`preprocess` → ``session.run`` → :meth:`postprocess`,
        merging tile detections with NMS when tiling is enabled.
        """

        if self._session is None or self._label_map is None:
            raise RuntimeError("Detector not loaded; call load() before infer()")

        prepared = self.preprocess(frame)

        if self.config.tile_size is None:
            raw = self._run_session(prepared)  # type: ignore[arg-type]
            dets = self.postprocess(raw, tile_metadata=None)
            logger.info("infer: %d detections", len(dets))
            return dets

        tensors, metas = prepared  # type: ignore[misc]
        all_dets: List[Dict] = []
        for tensor, meta in zip(tensors, metas):
            raw = self._run_session(tensor)
            all_dets.extend(self.postprocess(raw, tile_metadata=meta))
        merged = _nms(all_dets, self.config.iou_threshold)
        logger.info("infer (tiled): %d raw -> %d merged detections", len(all_dets), len(merged))
        return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def xyxy_to_xywh(box: Sequence[float]) -> List[int]:
    """Convert a corner box ``[x1, y1, x2, y2]`` to ``[x, y, w, h]`` (ints)."""

    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [int(round(x1)), int(round(y1)),
            int(round(max(0.0, x2 - x1))), int(round(max(0.0, y2 - y1)))]


def xywh_to_xyxy(box: Sequence[float]) -> List[int]:
    """Convert an ``[x, y, w, h]`` box to a corner box ``[x1, y1, x2, y2]`` (ints)."""

    x, y, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]


def _first_present(d: Dict, keys: Sequence[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    """Intersection-over-union of two ``[x, y, w, h]`` boxes."""

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = max(0, aw) * max(0, ah) + max(0, bw) * max(0, bh) - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Greedy per-label NMS over detection dicts (highest score wins)."""

    kept: List[Dict] = []
    for det in sorted(detections, key=lambda d: d["score"], reverse=True):
        if all(
            det["label"] != k["label"] or _iou(det["bbox"], k["bbox"]) < iou_threshold
            for k in kept
        ):
            kept.append(det)
    return kept
