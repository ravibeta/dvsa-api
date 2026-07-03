"""Unit tests for the ``urban_accident_example`` adapter's decision logic.

Loads the example adapter directly via the local loader and runs it against the
shipped ``sample_input.json``; asserts the deterministic accident detection and
the ``nominal`` path. No Django / network required.
"""

import json
import os

import pytest

from dvsa_api.reasoning.local_adapter_loader import load_local_adapter

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "custom_models", "reasoning", "urban_accident_example")


def _load_json(name):
    with open(os.path.join(_MODEL_DIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture()
def adapter():
    manifest = _load_json("manifest.json")
    return load_local_adapter(_MODEL_DIR, manifest)


def test_health_check_ok(adapter):
    assert adapter.health_check()["status"] == "ok"


def test_sample_input_detects_accident(adapter):
    context = _load_json("sample_input.json")
    out = adapter.predict(context)
    assert set(out) >= {"actions", "reasoning_trace", "metadata"}
    anomaly = out["actions"][0]
    assert anomaly["type"] == "anomaly"
    assert anomaly["label"] == "accident"
    assert anomaly["confidence"] == 0.95
    assert anomaly["bbox"] == [7.25, 5.0, 10.0, 10.0]
    assert set(anomaly["tracks"]) == {"veh-1", "veh-2"}
    # Explainable trace is populated.
    assert any("accident" in step for step in out["reasoning_trace"])


def test_matches_sample_output_actions(adapter):
    context = _load_json("sample_input.json")
    expected = _load_json("sample_output.json")
    out = adapter.predict(context)
    assert out["actions"] == expected["actions"]


def test_no_convergence_is_nominal(adapter):
    # Two tracks that never come close and never decelerate.
    context = {
        "query": "quiet street",
        "tracks": [
            {"id": "veh-1", "samples": [
                {"t": 0.0, "x": 0.0, "y": 0.0},
                {"t": 1.0, "x": 10.0, "y": 0.0},
                {"t": 2.0, "x": 20.0, "y": 0.0}]},
            {"id": "veh-2", "samples": [
                {"t": 0.0, "x": 0.0, "y": 500.0},
                {"t": 1.0, "x": 10.0, "y": 500.0},
                {"t": 2.0, "x": 20.0, "y": 500.0}]},
        ],
    }
    out = adapter.predict(context)
    assert out["actions"][0]["label"] == "nominal"


def test_single_track_is_nominal(adapter):
    out = adapter.predict({"tracks": [{"id": "veh-1", "samples": [
        {"t": 0.0, "x": 0.0, "y": 0.0}, {"t": 1.0, "x": 1.0, "y": 0.0}]}]})
    assert out["actions"][0]["label"] == "nominal"
