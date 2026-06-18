"""Integration tests for the custom ONNX detector.

These exercise the full ``preprocess -> session.run -> postprocess`` pipeline
with a *mocked* ``onnxruntime`` session, so they run without a real ``.onnx``
binary or the ``onnxruntime`` package installed. Only ``numpy`` and ``cv2``
(for the resize/colour-convert in preprocessing) are required.

Run with::

    pytest tests/test_custom_model_integration.py -q
"""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from custom_model.model_loader import (
    ModelConfig,
    create_detector,
    load_label_map,
    validate_model_file,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _make_named_mock(name):
    from unittest.mock import MagicMock

    m = MagicMock()
    m.name = name  # set as a real attribute (MagicMock(name=...) would NOT)
    return m


def make_fake_session(boxes=None, scores=None, labels=None, combined=None):
    """Build a MagicMock that mimics ``onnxruntime.InferenceSession``.

    By default returns the three-array layout ``[boxes, scores, labels]``.
    Pass ``combined`` to instead return a single ``(N, 6)`` array.
    """

    from unittest.mock import MagicMock

    if boxes is None:
        boxes = np.array([[0.1, 0.1, 0.5, 0.5]], dtype=np.float32)
    if scores is None:
        scores = np.array([0.95], dtype=np.float32)
    if labels is None:
        labels = np.array([1], dtype=np.int64)

    sess = MagicMock()

    def run(output_names, input_feed):
        if combined is not None:
            return [combined]
        return [boxes, scores, labels]

    sess.run.side_effect = run
    sess.get_inputs.return_value = [_make_named_mock("input")]
    sess.get_outputs.return_value = [
        _make_named_mock("boxes"),
        _make_named_mock("scores"),
        _make_named_mock("labels"),
    ]
    return sess


def _write_labels(tmp_path, mapping=None):
    mapping = mapping or {"0": "person", "1": "vehicle"}
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(mapping), encoding="utf-8")
    return str(p)


def _config(labels_path, **overrides):
    params = dict(
        onnx_path="fake.onnx",
        labels_path=labels_path,
        input_size=(640, 640),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    params.update(overrides)
    return ModelConfig(**params)


# --------------------------------------------------------------------------- #
# load_label_map / validate_model_file
# --------------------------------------------------------------------------- #


def test_load_label_map_coerces_keys_to_int(tmp_path):
    path = _write_labels(tmp_path, {"0": "person", "1": "vehicle", "2": "bicycle"})
    mapping = load_label_map(path)
    assert mapping == {0: "person", 1: "vehicle", 2: "bicycle"}
    assert all(isinstance(k, int) for k in mapping)


def test_load_label_map_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        load_label_map(str(tmp_path / "does_not_exist.json"))


def test_load_label_map_bad_class_id_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"person": "vehicle"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_label_map(str(p))


def test_validate_model_file_rejects_non_onnx(tmp_path):
    f = tmp_path / "model.bin"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_model_file(str(f))


# --------------------------------------------------------------------------- #
# infer() — happy paths
# --------------------------------------------------------------------------- #


def test_infer_returns_expected_labels(tmp_path):
    cfg = _config(_write_labels(tmp_path))
    detector = create_detector(cfg, session_factory=lambda p: make_fake_session())
    detector.load()

    frame = (np.ones((800, 1200, 3), dtype=np.uint8) * 255)
    dets = detector.infer(frame)

    assert isinstance(dets, list)
    assert len(dets) == 1
    assert dets[0]["label"] == "vehicle"
    assert 0.0 <= dets[0]["score"] <= 1.0
    assert len(dets[0]["bbox"]) == 4
    detector.close()


def test_infer_scales_bbox_to_original_frame(tmp_path):
    # Boxes are normalised [0,1]; expect them scaled to the 1200x800 frame.
    cfg = _config(_write_labels(tmp_path))
    detector = create_detector(cfg, session_factory=lambda p: make_fake_session())
    detector.load()

    frame = np.zeros((800, 1200, 3), dtype=np.uint8)
    dets = detector.infer(frame)

    # corners x:[120..600] y:[80..400] -> (x, y, w, h) = (120, 80, 480, 320)
    assert dets[0]["bbox"] == [120, 80, 480, 320]


def test_infer_handles_single_combined_output(tmp_path):
    cfg = _config(_write_labels(tmp_path))
    combined = np.array([[0.1, 0.1, 0.5, 0.5, 0.95, 1]], dtype=np.float32)
    detector = create_detector(
        cfg, session_factory=lambda p: make_fake_session(combined=combined)
    )
    detector.load()

    dets = detector.infer(np.zeros((800, 1200, 3), dtype=np.uint8))
    assert dets[0]["label"] == "vehicle"
    assert dets[0]["bbox"] == [120, 80, 480, 320]


def test_score_threshold_filters_low_confidence(tmp_path):
    cfg = _config(_write_labels(tmp_path), score_threshold=0.5)
    detector = create_detector(
        cfg,
        session_factory=lambda p: make_fake_session(
            scores=np.array([0.10], dtype=np.float32)
        ),
    )
    detector.load()
    dets = detector.infer(np.zeros((800, 1200, 3), dtype=np.uint8))
    assert dets == []


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_unknown_class_id_raises_valueerror(tmp_path):
    cfg = _config(_write_labels(tmp_path, {"0": "person", "1": "vehicle"}))
    detector = create_detector(
        cfg,
        session_factory=lambda p: make_fake_session(
            labels=np.array([99], dtype=np.int64)
        ),
    )
    detector.load()
    with pytest.raises(ValueError):
        detector.infer(np.zeros((800, 1200, 3), dtype=np.uint8))


def test_unexpected_output_shape_raises_runtimeerror(tmp_path):
    cfg = _config(_write_labels(tmp_path))
    detector = create_detector(cfg, session_factory=lambda p: make_fake_session())
    detector.load()
    detector.preprocess(np.zeros((10, 10, 3), dtype=np.uint8))  # sets orig shape
    with pytest.raises(RuntimeError):
        detector.postprocess([np.zeros((1, 2), dtype=np.float32)])


def test_infer_before_load_raises(tmp_path):
    cfg = _config(_write_labels(tmp_path))
    detector = create_detector(cfg, session_factory=lambda p: make_fake_session())
    with pytest.raises(RuntimeError):
        detector.infer(np.zeros((10, 10, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #


def test_tiled_inference_merges_with_nms(tmp_path):
    # Each tile yields the same normalised box; after mapping back to absolute
    # coords NMS should collapse overlapping duplicates per label.
    cfg = _config(
        _write_labels(tmp_path),
        input_size=(64, 64),
        tile_size=(400, 400),
        tile_overlap=0.0,
        iou_threshold=0.3,
    )
    detector = create_detector(cfg, session_factory=lambda p: make_fake_session())
    detector.load()

    dets = detector.infer(np.zeros((800, 1200, 3), dtype=np.uint8))
    # 2x3 = 6 tiles each produce one box; they sit in different tiles so NMS
    # keeps the non-overlapping ones.
    assert len(dets) >= 1
    assert all(d["label"] == "vehicle" for d in dets)
    assert all(len(d["bbox"]) == 4 for d in dets)
