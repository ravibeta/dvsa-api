"""Unit tests for ``custom_models.loader`` — ModelSpec / discover_model / labels.

These require no model binaries and no heavy runtimes; only the standard library
plus the package under test.

Run with::

    pytest tests/test_loader.py -q
"""

import json

import pytest

from custom_models.loader import (
    EXTENSION_FORMATS,
    ModelSpec,
    discover_model,
    load_label_map,
)


# --------------------------------------------------------------------------- #
# discover_model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path, expected_format",
    [
        ("/weights/model.onnx", "onnx"),
        ("/weights/detector.pt", "torch"),
        ("/weights/detector.pth", "torch"),
        ("/weights/scripted.torchscript", "torch"),
    ],
)
def test_discover_model_infers_format_from_extension(path, expected_format):
    spec = discover_model(path)
    assert spec.format == expected_format
    assert spec.path == path


@pytest.mark.parametrize(
    "path",
    ["/weights/yolov8x.pt", "/weights/visdrone-yolov5.onnx", "/weights/YOLO_best.pth"],
)
def test_discover_model_routes_yolo_filenames_to_yolo(path):
    # A "yolo" in the stem overrides the extension-derived format so the
    # Ultralytics adapter handles both .pt checkpoints and exported YOLO ONNX.
    assert discover_model(path).format == "yolo"


def test_discover_model_sets_id_from_basename_and_labels():
    spec = discover_model("/a/b/my_model.onnx", labels_file="/a/b/labels.json")
    assert spec.id == "my_model"
    assert spec.labels_file == "/a/b/labels.json"


def test_discover_model_unknown_extension_raises():
    with pytest.raises(ValueError):
        discover_model("/weights/model.weights")


def test_discover_model_empty_path_raises():
    with pytest.raises(ValueError):
        discover_model("")


def test_extension_formats_table_is_consistent():
    assert EXTENSION_FORMATS[".onnx"] == "onnx"
    assert EXTENSION_FORMATS[".pt"] == "torch"


# --------------------------------------------------------------------------- #
# ModelSpec
# --------------------------------------------------------------------------- #


def test_modelspec_normalises_fields():
    spec = ModelSpec(
        id="m",
        format="ONNX",
        input_size=[320, 320],
        tile_recommendation=[1024, 1024],
        capabilities=["Vehicle", "Person"],
    )
    assert spec.format == "onnx"
    assert spec.input_size == (320, 320)
    assert spec.tile_recommendation == (1024, 1024)
    assert spec.capabilities == ["vehicle", "person"]


def test_modelspec_from_catalog_parses_entry_and_resolves_base_dir():
    entry = {
        "id": "visdrone-yolov8x",
        "name": "VisDrone YOLOv8x",
        "format": "yolo",
        "source_url": "https://example/model",
        "artifact_filename": "visdrone-yolov8x.pt",
        "labels_file": "labels/visdrone.json",
        "input_size": [640, 640],
        "tile_recommendation": [1280, 1280],
        "task": "detection",
        "altitude": "medium",
        "capabilities": ["vehicle", "person"],
        "extra_field": "preserved",
    }
    spec = ModelSpec.from_catalog(entry, base_dir="/weights")
    assert spec.id == "visdrone-yolov8x"
    assert spec.format == "yolo"
    # os.path.join uses the platform separator, so assert on the basename only.
    assert spec.path.endswith("visdrone-yolov8x.pt")
    assert "weights" in spec.path
    assert spec.labels_file.endswith("visdrone.json")
    assert spec.tile_recommendation == (1280, 1280)
    assert spec.altitude == "medium"
    assert spec.metadata.get("extra_field") == "preserved"


def test_modelspec_from_catalog_accepts_wxh_strings_and_null_tile():
    entry = {"id": "m", "format": "onnx", "input_size": "320x240", "tile_recommendation": None}
    spec = ModelSpec.from_catalog(entry)
    assert spec.input_size == (320, 240)
    assert spec.tile_recommendation is None


# --------------------------------------------------------------------------- #
# load_label_map (re-exported from custom_model)
# --------------------------------------------------------------------------- #


def test_load_label_map_coerces_keys_to_int(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"0": "person", "1": "vehicle"}), encoding="utf-8")
    mapping = load_label_map(str(p))
    assert mapping == {0: "person", 1: "vehicle"}


def test_load_label_map_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        load_label_map(str(tmp_path / "nope.json"))
