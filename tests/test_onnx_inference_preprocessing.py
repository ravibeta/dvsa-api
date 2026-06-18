"""Unit tests for ``CustomONNXDetector.preprocess`` (no ONNX session needed).

These validate the resize, BGR->RGB swap, normalisation and tensor layout, plus
the optional tiling behaviour. They use small synthetic images so the asserts
are exact.
"""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from custom_model.model_loader import ModelConfig
from custom_model.onnx_inference import CustomONNXDetector, TileMeta


def _detector(tmp_path, **overrides):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"0": "person", "1": "vehicle"}), encoding="utf-8")
    params = dict(
        onnx_path="fake.onnx",
        labels_path=str(labels),
        input_size=(8, 8),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    params.update(overrides)
    return CustomONNXDetector(ModelConfig(**params))


def test_preprocess_shape_and_dtype(tmp_path):
    det = _detector(tmp_path, input_size=(16, 12))  # (width, height)
    frame = np.ones((40, 50, 3), dtype=np.uint8) * 255
    tensor = det.preprocess(frame)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 12, 16)  # (N, C, H, W)
    assert tensor.dtype == np.float32


def test_preprocess_scales_to_unit_range(tmp_path):
    # mean=0, std=1 -> pixels map linearly to [0, 1]; all-255 frame -> all 1.0.
    det = _detector(tmp_path)
    frame = np.ones((8, 8, 3), dtype=np.uint8) * 255
    tensor = det.preprocess(frame)
    assert np.allclose(tensor, 1.0)


def test_preprocess_applies_mean_std(tmp_path):
    det = _detector(tmp_path, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    white = det.preprocess(np.ones((8, 8, 3), dtype=np.uint8) * 255)
    black = det.preprocess(np.zeros((8, 8, 3), dtype=np.uint8))
    # (1.0 - 0.5)/0.5 = 1.0 ; (0.0 - 0.5)/0.5 = -1.0
    assert np.allclose(white, 1.0)
    assert np.allclose(black, -1.0)


def test_preprocess_converts_bgr_to_rgb(tmp_path):
    det = _detector(tmp_path)
    # Fill a uniform BGR image: B=10, G=20, R=30.
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 10  # B
    frame[:, :, 1] = 20  # G
    frame[:, :, 2] = 30  # R
    tensor = det.preprocess(frame)
    # Channel order after BGR->RGB is R, G, B.
    assert np.allclose(tensor[0, 0], 30 / 255.0, atol=1e-3)  # R
    assert np.allclose(tensor[0, 1], 20 / 255.0, atol=1e-3)  # G
    assert np.allclose(tensor[0, 2], 10 / 255.0, atol=1e-3)  # B


def test_preprocess_rejects_bad_frame(tmp_path):
    det = _detector(tmp_path)
    with pytest.raises(ValueError):
        det.preprocess(np.ones((8, 8), dtype=np.uint8))  # missing channel dim
    with pytest.raises(ValueError):
        det.preprocess(None)


def test_preprocess_tiling_returns_tensors_and_metadata(tmp_path):
    det = _detector(tmp_path, input_size=(8, 8), tile_size=(100, 100), tile_overlap=0.0)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    tensors, metas = det.preprocess(frame)
    assert isinstance(tensors, list) and isinstance(metas, list)
    assert len(tensors) == len(metas) == 6  # 2 rows x 3 cols of 100px tiles
    assert all(t.shape == (1, 3, 8, 8) for t in tensors)
    assert all(isinstance(m, TileMeta) for m in metas)
    # Tiles cover the frame from the origin.
    assert metas[0] == TileMeta(x_off=0, y_off=0, width=100, height=100)


def test_preprocess_tiling_with_overlap_produces_more_tiles(tmp_path):
    no_overlap = _detector(tmp_path, tile_size=(100, 100), tile_overlap=0.0)
    overlap = _detector(tmp_path, tile_size=(100, 100), tile_overlap=0.5)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    _, metas_none = no_overlap.preprocess(frame)
    _, metas_over = overlap.preprocess(frame)
    assert len(metas_over) > len(metas_none)
