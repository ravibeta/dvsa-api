"""PyTorch adapter — TorchScript / ``.pt`` detection models with SAHI-style tiling.

Loads a TorchScript module (``torch.jit.load``) — or any model produced by an
injected ``model_loader`` — runs it over a BGR frame and returns the common
detection dicts ``{"label","score","bbox":[x,y,w,h]}`` in original-frame pixels.

Output layouts handled
----------------------
* torchvision-style: a list/tuple of ``{"boxes":(N,4), "labels":(N,), "scores":(N,)}``.
* a ``dict`` with ``boxes``/``labels``/``scores`` keys.
* a 3-tuple ``(boxes, scores, labels)``.
* a single ``(N, 6)`` array of ``[x1, y1, x2, y2, score, cls]``.

Boxes are assumed to be in the model's **input-pixel** space (torchvision
convention); they are rescaled back to the source frame using the spec's
``input_size``. With ``spec.tile_recommendation`` set, the frame is split into
overlapping tiles (SAHI-style), each inferred independently and merged with NMS.

The ``model_loader`` and ``to_tensor`` hooks are injectable so the adapter can be
unit-tested by mocking ``torch.jit.load`` (no real ``torch`` install required).

CLI smoke test::

    python -m custom_models.adapters.torch_adapter model.torchscript label_map.json image.jpg
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..loader import ModelSpec, load_label_map
from ..postprocess import nms, to_detections
from ..registry import register

logger = logging.getLogger(__name__)


@register("torch")
class TorchAdapter:
    """Run a TorchScript / ``.pt`` detection model described by a :class:`ModelSpec`.

    Parameters
    ----------
    spec:
        The :class:`ModelSpec`; ``spec.path`` points at a TorchScript/``.pt``
        artifact, ``spec.labels_file`` at a ``label_map.json``.
    model_loader:
        Optional callable ``(path) -> model`` used by :meth:`load`. Defaults to
        ``torch.jit.load``. Inject a stub to mock the model in tests.
    score_threshold:
        Minimum score to keep.
    tile_overlap:
        Fractional overlap between tiles in ``[0, 1)`` when tiling is enabled.
    iou_threshold:
        IoU threshold for merging tiled detections with NMS.
    """

    format = "torch"

    def __init__(
        self,
        spec: ModelSpec,
        *,
        model_loader: Optional[Callable[[str], object]] = None,
        score_threshold: float = 0.0,
        tile_overlap: float = 0.2,
        iou_threshold: float = 0.5,
    ) -> None:
        self.spec = spec
        self._model_loader = model_loader
        self._score_threshold = score_threshold
        self._tile_overlap = tile_overlap
        self._iou_threshold = iou_threshold
        self._model = None
        self._label_map: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------ #
    def load(self) -> "TorchAdapter":
        """Load the TorchScript model and the label map."""

        if self.spec.labels_file:
            self._label_map = load_label_map(self.spec.labels_file)
        else:
            self._label_map = {}

        loader = self._model_loader or self._default_loader
        self._model = loader(self.spec.path)
        # Put the model in eval mode when it supports it (real torch modules do).
        eval_fn = getattr(self._model, "eval", None)
        if callable(eval_fn):
            try:
                eval_fn()
            except Exception:  # pragma: no cover - defensive for stubs
                pass
        logger.info("TorchAdapter loaded model %s", self.spec.id)
        return self

    @staticmethod
    def _default_loader(path: str):  # pragma: no cover - requires torch + a real model
        import torch

        return torch.jit.load(path)

    def close(self) -> None:
        """Drop the model reference. Safe to call repeatedly."""

        self._model = None

    # ------------------------------------------------------------------ #
    def _to_tensor(self, region: np.ndarray):
        """Resize ``region`` to the model input and return a ``(1, C, H, W)`` tensor.

        Returns a ``torch.Tensor`` when ``torch`` is importable, otherwise the
        raw ``numpy`` array (sufficient for mocked models in tests).
        """

        import cv2

        in_w, in_h = self.spec.input_size
        resized = cv2.resize(region, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        try:
            import torch

            return torch.from_numpy(chw)
        except Exception:
            return chw

    def _run(self, tensor):
        """Call the model, returning its raw output."""

        return self._model(tensor)  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    def infer(self, frame: np.ndarray) -> List[Dict]:
        """Run the model over ``frame`` and return detection dicts."""

        if self._model is None or self._label_map is None:
            raise RuntimeError("TorchAdapter not loaded; call load() first")
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be an HxWx3 BGR numpy array")

        h, w = frame.shape[:2]

        if self.spec.tile_recommendation is None:
            raw = self._run(self._to_tensor(frame))
            return self._decode(raw, frame_hw=(h, w), region=(0, 0, w, h))

        all_dets: List[Dict] = []
        for region_img, region in _generate_tiles(
            frame, self.spec.tile_recommendation, self._tile_overlap
        ):
            raw = self._run(self._to_tensor(region_img))
            all_dets.extend(self._decode(raw, frame_hw=(h, w), region=region))
        merged = nms(all_dets, self._iou_threshold)
        logger.info("TorchAdapter infer (tiled): %d -> %d", len(all_dets), len(merged))
        return merged

    def _decode(self, raw, *, frame_hw, region) -> List[Dict]:
        boxes, scores, labels = _parse_torch_output(raw)
        return to_detections(
            boxes,
            scores,
            labels,
            self._label_map,
            frame_hw=frame_hw,
            region=region,
            model_input_size=self.spec.input_size,
            score_threshold=self._score_threshold,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_array(value) -> np.ndarray:
    """Convert a torch tensor / list / ndarray to a numpy array (CPU)."""

    if value is None:
        return np.empty((0,))
    detach = getattr(value, "detach", None)
    if callable(detach):  # torch.Tensor
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        npfn = getattr(value, "numpy", None)
        if callable(npfn):
            return np.asarray(npfn())
    return np.asarray(value)


def _parse_torch_output(raw) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise assorted torch detection outputs into ``(boxes, scores, labels)``."""

    # torchvision returns a list of per-image dicts; take the first image.
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
        raw = raw[0]

    if isinstance(raw, dict):
        boxes = _as_array(raw.get("boxes"))
        scores = _as_array(raw.get("scores"))
        labels = _as_array(raw.get("labels"))
        return boxes.reshape(-1, 4) if boxes.size else boxes.reshape(0, 4), \
            scores.reshape(-1), labels.reshape(-1)

    # 3-tuple (boxes, scores, labels)
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        boxes, scores, labels = _as_array(raw[0]), _as_array(raw[1]), _as_array(raw[2])
        return boxes.reshape(-1, 4) if boxes.size else boxes.reshape(0, 4), \
            scores.reshape(-1), labels.reshape(-1)

    # single combined (N, 6) array [x1,y1,x2,y2,score,cls]
    arr = _as_array(raw[0] if isinstance(raw, (list, tuple)) and len(raw) == 1 else raw)
    arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(1, -1)
    if arr.shape[-1] >= 6:
        return arr[:, 0:4], arr[:, 4].reshape(-1), arr[:, 5].reshape(-1)

    raise RuntimeError(
        f"Unsupported torch output layout with shape {getattr(arr, 'shape', None)}; "
        "expected per-image dicts, (boxes,scores,labels), or an (N,6) array"
    )


def _generate_tiles(
    frame: np.ndarray, tile_size: Tuple[int, int], overlap: float
) -> Sequence[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Yield ``(region_image, (x_off, y_off, w, h))`` covering ``frame`` with overlap."""

    h, w = frame.shape[:2]
    tile_w, tile_h = tile_size
    step_x = max(1, int(tile_w * (1.0 - overlap)))
    step_y = max(1, int(tile_h * (1.0 - overlap)))
    out = []
    for y0 in range(0, max(1, h - 1), step_y):
        for x0 in range(0, max(1, w - 1), step_x):
            x1, y1 = min(x0 + tile_w, w), min(y0 + tile_h, h)
            region_img = frame[y0:y1, x0:x1]
            if region_img.size == 0:
                continue
            out.append((region_img, (x0, y0, x1 - x0, y1 - y0)))
    return out


def _smoke_test(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    import cv2

    from ..loader import discover_model

    parser = argparse.ArgumentParser(description="Torch adapter smoke test")
    parser.add_argument("model", help="path to TorchScript/.pt model")
    parser.add_argument("labels", help="path to label_map.json")
    parser.add_argument("image", help="path to an image file")
    args = parser.parse_args(argv)

    spec = discover_model(args.model, labels_file=args.labels)
    spec.format = "torch"
    detector = TorchAdapter(spec).load()
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}")
        return 2
    for d in detector.infer(frame):
        print(f"  {d['label']:>12}  {d['score']:.3f}  bbox={d['bbox']}")
    detector.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_smoke_test())
