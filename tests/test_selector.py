"""Unit tests for ``custom_models.selector.ModelSelector``.

Covers catalog loading, id lookup, and query → best-fit ranking over a small
in-memory catalog plus the real bundled ``models_catalog.json``.

Run with::

    pytest tests/test_selector.py -q
"""

import json

import pytest

from custom_models.selector import ModelSelector, SelectionQuery


# A compact, deterministic catalog for ranking assertions.
_CATALOG = {
    "models": [
        {
            "id": "small-coco",
            "format": "yolo",
            "input_size": [640, 640],
            "tile_recommendation": None,
            "task": "detection",
            "altitude": "low",
            "capabilities": ["person", "car", "vehicle"],
        },
        {
            "id": "big-aerial",
            "format": "torch",
            "input_size": [1024, 1024],
            "tile_recommendation": [1024, 1024],
            "task": "detection",
            "altitude": "high",
            "capabilities": ["vehicle", "ship", "plane"],
        },
        {
            "id": "visdrone",
            "format": "yolo",
            "input_size": [640, 640],
            "tile_recommendation": [1280, 1280],
            "task": "detection",
            "altitude": "medium",
            "capabilities": ["person", "vehicle", "bicycle"],
        },
    ]
}


@pytest.fixture
def selector():
    return ModelSelector.from_catalog(_CATALOG)


# --------------------------------------------------------------------------- #
# Construction / lookup
# --------------------------------------------------------------------------- #


def test_from_file_loads_catalog(tmp_path):
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(_CATALOG), encoding="utf-8")
    sel = ModelSelector.from_file(str(p))
    assert {s.id for s in sel.specs} == {"small-coco", "big-aerial", "visdrone"}


def test_from_file_missing_raises(tmp_path):
    with pytest.raises(ValueError):
        ModelSelector.from_file(str(tmp_path / "nope.json"))


def test_get_by_id(selector):
    assert selector.get("visdrone").format == "yolo"


def test_get_unknown_id_raises(selector):
    with pytest.raises(KeyError):
        selector.get("does-not-exist")


# --------------------------------------------------------------------------- #
# Ranking / selection
# --------------------------------------------------------------------------- #


def test_select_prefers_class_coverage(selector):
    # All three list "vehicle"; only big-aerial covers ship+plane.
    spec = selector.select(classes=["ship", "plane"])
    assert spec.id == "big-aerial"


def test_select_high_res_prefers_tiling_model(selector):
    # 4K source + aerial classes + high altitude -> big-aerial wins decisively.
    spec = selector.select(
        classes=["vehicle"], altitude="high", resolution=(3840, 2160)
    )
    assert spec.id == "big-aerial"


def test_select_low_altitude_small_frame_prefers_coco(selector):
    spec = selector.select(
        classes=["person", "car"], altitude="low", resolution=(1280, 720)
    )
    assert spec.id == "small-coco"


def test_rank_is_sorted_descending_and_deterministic(selector):
    ranked = selector.rank(SelectionQuery(classes=["person", "vehicle"]))
    scores = [score for score, _ in ranked]
    assert scores == sorted(scores, reverse=True)
    # Two yolo models fully cover {person, vehicle}; tie breaks on id ascending.
    top_two_ids = {spec.id for _, spec in ranked[:2]}
    assert top_two_ids == {"small-coco", "visdrone"}


def test_select_accepts_query_object_or_kwargs(selector):
    by_obj = selector.select(SelectionQuery(classes=["ship"]))
    by_kwargs = selector.select(classes=["ship"])
    assert by_obj.id == by_kwargs.id == "big-aerial"


def test_empty_catalog_select_raises():
    with pytest.raises(ValueError):
        ModelSelector(specs=[]).select(classes=["person"])


# --------------------------------------------------------------------------- #
# Bundled catalog
# --------------------------------------------------------------------------- #


def test_default_catalog_loads_and_is_selectable():
    sel = ModelSelector.default()
    assert len(sel.specs) >= 5
    spec = sel.select(
        classes=["vehicle", "person"], altitude="high", resolution=(3840, 2160)
    )
    # A high-altitude, tiling-capable model should win for 4K aerial vehicle/person.
    assert spec.tile_recommendation is not None
    assert spec.format in {"yolo", "torch", "onnx"}
