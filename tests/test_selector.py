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


def test_default_catalog_has_at_least_twelve_curated_models():
    # curate_model_2.md acceptance criterion #3: >= 12 curated entries.
    sel = ModelSelector.default()
    assert len(sel.specs) >= 12


def test_default_catalog_entries_are_well_formed():
    sel = ModelSelector.default()
    ids = [s.id for s in sel.specs]
    assert len(ids) == len(set(ids)), "catalog ids must be unique"
    for spec in sel.specs:
        # Every entry must cite an authoritative http(s) source_url.
        assert spec.source_url.startswith("http"), f"{spec.id} missing source_url"
        # Every format must map to a registered adapter.
        assert spec.format in {"yolo", "torch", "onnx"}, f"{spec.id} bad format"
        assert spec.capabilities, f"{spec.id} has no capabilities"


def test_default_catalog_covers_ship_and_building_specialists():
    # The expanded catalog adds maritime + building-footprint coverage; the
    # selector should pick a model that actually covers the requested class.
    sel = ModelSelector.default()
    ids = {s.id for s in sel.specs}
    assert {"hrsc2016-ship", "spacenet-buildings"} <= ids
    assert "ship" in sel.select(classes=["ship"]).capabilities
    assert "building" in sel.select(classes=["building"]).capabilities
