"""Unit tests for the ``custom_models`` adapters with all runtimes mocked.

* ONNX  — mocked two ways: an injected ``session_factory`` *and* a fake
  ``onnxruntime`` module patched into ``sys.modules`` (real load path).
* Torch — injected ``model_loader`` *and* a patched ``torch.jit.load``.
* YOLO  — injected ``model_factory`` returning fake Ultralytics ``Results``.

No real model binaries, ``onnxruntime``, ``torch`` or ``ultralytics`` install is
required — only ``numpy`` and ``cv2`` (for the resize in preprocessing).

Run with::

    pytest tests/test_adapters.py -q
"""

import json
import sys
import types
from unittest import mock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from custom_models import available_formats, get_detector
from custom_models.adapters.onnx_adapter import ONNXAdapter
from custom_models.adapters.torch_adapter import TorchAdapter, _parse_torch_output
from custom_models.adapters.yolo_adapter import YOLOAdapter
from custom_models.loader import ModelSpec


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #


def _labels(tmp_path, mapping=None):
    mapping = mapping or {"0": "person", "1": "vehicle"}
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(mapping), encoding="utf-8")
    return str(p)


def _named(name):
    m = mock.MagicMock()
    m.name = name
    return m


def make_fake_onnx_session(boxes=None, scores=None, labels=None):
    """A MagicMock mimicking onnxruntime.InferenceSession (3-array layout)."""

    boxes = np.array([[0.1, 0.1, 0.5, 0.5]], dtype=np.float32) if boxes is None else boxes
    scores = np.array([0.95], dtype=np.float32) if scores is None else scores
    labels = np.array([1], dtype=np.int64) if labels is None else labels

    sess = mock.MagicMock()
    sess.run.side_effect = lambda output_names, feed: [boxes, scores, labels]
    sess.get_inputs.return_value = [_named("input")]
    sess.get_outputs.return_value = [_named("boxes"), _named("scores"), _named("labels")]
    return sess


class _FakeTorchModel:
    """Callable returning torchvision-style per-image detection dicts."""

    def __init__(self, boxes, scores, labels):
        self._out = [{"boxes": boxes, "scores": scores, "labels": labels}]

    def __call__(self, tensor):
        return self._out


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class _FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLOModel:
    def __init__(self, xyxy, conf, cls, names):
        self._results = [_FakeResults(_FakeBoxes(xyxy, conf, cls))]
        self.names = names

    def __call__(self, frame, conf=None, verbose=None):
        return self._results


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_exposes_builtin_formats():
    formats = available_formats()
    assert {"onnx", "torch", "yolo"} <= set(formats)


def test_get_detector_builds_adapter_for_spec_format(tmp_path):
    spec = ModelSpec(id="m", format="onnx", path="fake.onnx", labels_file=_labels(tmp_path))
    det = get_detector(spec, session_factory=lambda p: make_fake_onnx_session())
    assert isinstance(det, ONNXAdapter)


# --------------------------------------------------------------------------- #
# ONNX adapter
# --------------------------------------------------------------------------- #


def test_onnx_adapter_infer_via_injected_session(tmp_path):
    spec = ModelSpec(
        id="m", format="onnx", path="fake.onnx",
        labels_file=_labels(tmp_path), input_size=(640, 640),
    )
    det = ONNXAdapter(spec, session_factory=lambda p: make_fake_onnx_session()).load()

    dets = det.infer(np.zeros((800, 1200, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0]["label"] == "vehicle"
    assert 0.0 <= dets[0]["score"] <= 1.0
    # Normalised box [0.1,0.1,0.5,0.5] over a 1200x800 frame -> (120,80,480,320).
    assert dets[0]["bbox"] == [120, 80, 480, 320]
    det.close()
    det.close()  # idempotent


def test_onnx_adapter_with_mocked_onnxruntime_module(tmp_path):
    """Exercise the *real* load path with a fake onnxruntime in sys.modules."""

    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"not-a-real-onnx")  # only needs to exist + end in .onnx
    labels = _labels(tmp_path)

    fake_session = make_fake_onnx_session()
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = mock.MagicMock(return_value=fake_session)

    spec = ModelSpec(id="m", format="onnx", path=str(model_file), labels_file=labels)
    with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        det = ONNXAdapter(spec).load()
        dets = det.infer(np.zeros((800, 1200, 3), dtype=np.uint8))

    fake_ort.InferenceSession.assert_called_once()
    assert dets[0]["label"] == "vehicle"
    assert dets[0]["bbox"] == [120, 80, 480, 320]


def test_onnx_adapter_requires_labels(tmp_path):
    spec = ModelSpec(id="m", format="onnx", path="fake.onnx", labels_file=None)
    with pytest.raises(ValueError):
        ONNXAdapter(spec, session_factory=lambda p: make_fake_onnx_session()).load()


def test_onnx_adapter_infer_before_load_raises(tmp_path):
    spec = ModelSpec(id="m", format="onnx", path="fake.onnx", labels_file=_labels(tmp_path))
    with pytest.raises(RuntimeError):
        ONNXAdapter(spec).infer(np.zeros((10, 10, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Torch adapter
# --------------------------------------------------------------------------- #


def test_torch_adapter_infer_via_injected_loader(tmp_path):
    boxes = np.array([[10, 20, 110, 220]], dtype=np.float32)  # input-pixel space
    scores = np.array([0.9], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)
    model = _FakeTorchModel(boxes, scores, labels)

    spec = ModelSpec(
        id="m", format="torch", path="fake.pt",
        labels_file=_labels(tmp_path), input_size=(640, 640),
    )
    det = TorchAdapter(spec, model_loader=lambda p: model).load()

    # Square frame at the model input size -> 1:1 box scaling.
    dets = det.infer(np.zeros((640, 640, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0]["label"] == "vehicle"
    assert dets[0]["bbox"] == [10, 20, 100, 200]
    det.close()


def test_torch_adapter_with_patched_torch_jit_load(tmp_path):
    """Exercise the default loader by patching torch.jit.load."""

    boxes = np.array([[0, 0, 320, 320]], dtype=np.float32)
    model = _FakeTorchModel(boxes, np.array([0.8]), np.array([0]))

    fake_torch = types.ModuleType("torch")
    fake_torch.jit = types.SimpleNamespace(load=mock.MagicMock(return_value=model))

    spec = ModelSpec(id="m", format="torch", path="fake.pt",
                     labels_file=_labels(tmp_path), input_size=(640, 640))
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        det = TorchAdapter(spec).load()
        dets = det.infer(np.zeros((640, 640, 3), dtype=np.uint8))

    fake_torch.jit.load.assert_called_once()
    assert dets[0]["label"] == "person"
    assert dets[0]["bbox"] == [0, 0, 320, 320]


def test_torch_adapter_score_threshold_filters(tmp_path):
    model = _FakeTorchModel(
        np.array([[10, 20, 110, 220]], dtype=np.float32),
        np.array([0.10], dtype=np.float32),
        np.array([1]),
    )
    spec = ModelSpec(id="m", format="torch", path="fake.pt",
                     labels_file=_labels(tmp_path), input_size=(640, 640))
    det = TorchAdapter(spec, model_loader=lambda p: model, score_threshold=0.5).load()
    assert det.infer(np.zeros((640, 640, 3), dtype=np.uint8)) == []


def test_torch_adapter_tiling_runs_per_tile_and_merges(tmp_path):
    # Each tile returns one normalised-ish box; with tiling enabled the adapter
    # should call the model once per tile and return >= 1 merged detection.
    box = np.array([[10, 10, 60, 60]], dtype=np.float32)
    model = mock.MagicMock(side_effect=lambda t: [{"boxes": box,
                                                   "scores": np.array([0.9]),
                                                   "labels": np.array([1])}])
    spec = ModelSpec(
        id="m", format="torch", path="fake.pt", labels_file=_labels(tmp_path),
        input_size=(64, 64), tile_recommendation=(400, 400),
    )
    det = TorchAdapter(spec, model_loader=lambda p: model, tile_overlap=0.0).load()
    dets = det.infer(np.zeros((800, 1200, 3), dtype=np.uint8))
    assert model.call_count >= 2  # multiple tiles
    assert all(d["label"] == "vehicle" for d in dets)
    assert all(len(d["bbox"]) == 4 for d in dets)


def test_parse_torch_output_supports_combined_array():
    arr = np.array([[10, 20, 110, 220, 0.9, 1]], dtype=np.float32)
    boxes, scores, labels = _parse_torch_output(arr)
    assert boxes.shape == (1, 4)
    assert scores.tolist() == [0.9000000357627869] or scores[0] == pytest.approx(0.9)
    assert labels[0] == 1


# --------------------------------------------------------------------------- #
# YOLO adapter
# --------------------------------------------------------------------------- #


def test_yolo_adapter_infer_uses_model_names(tmp_path):
    xyxy = np.array([[10, 20, 110, 220]], dtype=np.float32)  # original-frame px
    conf = np.array([0.9], dtype=np.float32)
    cls = np.array([1], dtype=np.float32)
    model = _FakeYOLOModel(xyxy, conf, cls, names={0: "person", 1: "vehicle"})

    spec = ModelSpec(id="m", format="yolo", path="fake.pt", labels_file=None)
    det = YOLOAdapter(spec, model_factory=lambda p: model).load()

    dets = det.infer(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0]["label"] == "vehicle"
    # YOLO boxes are already in frame pixels -> no rescale, just xywh convert.
    assert dets[0]["bbox"] == [10, 20, 100, 200]
    det.close()


def test_yolo_adapter_score_threshold_filters(tmp_path):
    model = _FakeYOLOModel(
        np.array([[10, 20, 110, 220]], dtype=np.float32),
        np.array([0.10], dtype=np.float32),
        np.array([1], dtype=np.float32),
        names={0: "person", 1: "vehicle"},
    )
    spec = ModelSpec(id="m", format="yolo", path="fake.pt")
    det = YOLOAdapter(spec, model_factory=lambda p: model, score_threshold=0.25).load()
    assert det.infer(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_yolo_adapter_prefers_explicit_labels_file(tmp_path):
    labels = _labels(tmp_path, {"0": "ped", "1": "truck"})
    model = _FakeYOLOModel(
        np.array([[1, 2, 3, 4]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([1], dtype=np.float32),
        names={0: "person", 1: "vehicle"},  # should be overridden by labels file
    )
    spec = ModelSpec(id="m", format="yolo", path="fake.pt", labels_file=labels)
    det = YOLOAdapter(spec, model_factory=lambda p: model, score_threshold=0.0).load()
    dets = det.infer(np.zeros((480, 640, 3), dtype=np.uint8))
    assert dets[0]["label"] == "truck"
