"""Tests for the dvsa-api wiring of the custom ONNX detector.

These cover the env -> ModelConfig parsing, the adapter-dict -> Detection mapping
(crucially the ``[x, y, w, h]`` contract shared with
``apps.analytics.routines.base.Detection``), and the registered routine's
dispatch / error behaviour. No Django, ONNX runtime or real model is needed.
"""

import numpy as np
import pytest

from apps.analytics.routines import get_routine, run_frame_routine
from apps.analytics.routines import custom_onnx
from apps.analytics.routines.base import Detection


# --------------------------------------------------------------------------- #
# Env -> ModelConfig
# --------------------------------------------------------------------------- #


def test_build_config_from_env_unconfigured_returns_none():
    assert custom_onnx.build_config_from_env(env={}) is None
    # Only one of the two required paths is not enough.
    assert custom_onnx.build_config_from_env(env={"CUSTOM_MODEL_ONNX_PATH": "m.onnx"}) is None


def test_build_config_from_env_parses_values():
    env = {
        "CUSTOM_MODEL_ONNX_PATH": "/models/m.onnx",
        "CUSTOM_MODEL_LABELS_PATH": "/models/labels.json",
        "CUSTOM_MODEL_INPUT_SIZE": "512x384",
        "CUSTOM_MODEL_MEAN": "0.1,0.2,0.3",
        "CUSTOM_MODEL_STD": "0.4,0.5,0.6",
        "CUSTOM_MODEL_TILE_SIZE": "1024x1024",
        "CUSTOM_MODEL_TILE_OVERLAP": "0.25",
        "CUSTOM_MODEL_SCORE_THRESHOLD": "0.4",
    }
    cfg = custom_onnx.build_config_from_env(env=env)
    assert cfg is not None
    assert cfg.onnx_path == "/models/m.onnx"
    assert cfg.input_size == (512, 384)
    assert cfg.mean == (0.1, 0.2, 0.3)
    assert cfg.std == (0.4, 0.5, 0.6)
    assert cfg.tile_size == (1024, 1024)
    assert cfg.tile_overlap == 0.25
    assert cfg.score_threshold == 0.4


def test_build_config_from_env_defaults_when_minimal():
    cfg = custom_onnx.build_config_from_env(
        env={
            "CUSTOM_MODEL_ONNX_PATH": "/m.onnx",
            "CUSTOM_MODEL_LABELS_PATH": "/l.json",
        }
    )
    assert cfg.input_size == (640, 640)
    assert cfg.tile_size is None  # tiling disabled by default


def test_build_config_from_env_rejects_bad_size():
    with pytest.raises(ValueError):
        custom_onnx.build_config_from_env(
            env={
                "CUSTOM_MODEL_ONNX_PATH": "/m.onnx",
                "CUSTOM_MODEL_LABELS_PATH": "/l.json",
                "CUSTOM_MODEL_INPUT_SIZE": "640-640",
            }
        )


# --------------------------------------------------------------------------- #
# Adapter dict -> Detection / RoutineResult
# --------------------------------------------------------------------------- #


def test_detection_from_dict_maps_xywh():
    det = custom_onnx._detection_from_dict(
        {"label": "vehicle", "score": 0.9, "bbox": [120, 80, 480, 320]}
    )
    assert isinstance(det, Detection)
    assert det.bbox == (120, 80, 480, 320)         # (x, y, w, h) preserved
    assert det.centroid == (120 + 240.0, 80 + 160.0)  # centre of the box
    assert det.area == 480 * 320
    assert det.label == "vehicle"
    assert det.score == 0.9


def test_build_routine_result_envelope():
    out = custom_onnx.build_routine_result(
        [
            {"label": "vehicle", "score": 0.9, "bbox": [10, 10, 20, 20]},
            {"label": "person", "score": 0.8, "bbox": [50, 50, 5, 5]},
        ]
    )
    assert out["routine"] == "custom_onnx_detection"
    assert out["summary"]["count"] == 2
    assert out["summary"]["labels"] == ["person", "vehicle"]
    assert len(out["detections"]) == 2
    # Detection.to_dict() emits bbox as a JSON list of ints.
    assert out["detections"][0]["bbox"] == [10, 10, 20, 20]


# --------------------------------------------------------------------------- #
# Registered routine dispatch
# --------------------------------------------------------------------------- #


def test_routine_is_registered():
    spec = get_routine("custom_onnx_detection")
    assert spec.level == "frame"


class _FakeDetector:
    def infer(self, frame):
        return [{"label": "vehicle", "score": 0.95, "bbox": [1, 2, 3, 4]}]


def test_routine_dispatches_to_detector(monkeypatch):
    monkeypatch.setattr(custom_onnx, "get_custom_detector", lambda: _FakeDetector())
    out = run_frame_routine("custom_onnx_detection", np.zeros((10, 10, 3), np.uint8))
    assert out["routine"] == "custom_onnx_detection"
    assert out["detections"][0]["label"] == "vehicle"
    assert out["detections"][0]["bbox"] == [1, 2, 3, 4]


def test_routine_raises_when_unconfigured(monkeypatch):
    monkeypatch.setattr(custom_onnx, "get_custom_detector", lambda: None)
    with pytest.raises(RuntimeError):
        run_frame_routine("custom_onnx_detection", np.zeros((10, 10, 3), np.uint8))
