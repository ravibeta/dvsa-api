"""API tests for the generic pluggable-reasoning router.

Drives the DRF endpoints end-to-end with the test client against the on-disk
example model. No DB models are touched (no ``django_db`` marker needed) and the
Azure adapter is never invoked (fallback disabled by default in CI).
"""

from __future__ import annotations

import json
import os

import pytest
from rest_framework.test import APIClient

from dvsa_api.reasoning import registry

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "custom_models", "reasoning", "urban_accident_example")


def _sample_input():
    with open(os.path.join(_MODEL_DIR, "sample_input.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def client():
    registry.discover_models(force=True)
    return APIClient()


def test_infer_by_name_returns_200_with_telemetry(client):
    resp = client.post(
        "/api/reasoning/urban_accident_example/infer",
        _sample_input(), format="json")
    assert resp.status_code == 200, resp.content
    data = resp.json()
    # Adapter output verbatim ...
    assert {"actions", "reasoning_trace", "metadata"} <= set(data)
    assert data["actions"][0]["label"] == "accident"
    # ... plus server telemetry.
    assert data["selected_model"] == "urban_accident_example"
    assert "request_id" in data
    assert isinstance(data["server_latency_ms"], int)


def test_infer_auto_selects_model(client):
    resp = client.post(
        "/api/reasoning/infer?policy=latency_optimized",
        _sample_input(), format="json")
    assert resp.status_code == 200, resp.content
    data = resp.json()
    assert data["selected_model"] == "urban_accident_example"
    assert {"actions", "metadata"} <= set(data)


def test_unknown_model_returns_structured_error(client):
    resp = client.post(
        "/api/reasoning/nonexistent_model/infer", {"tracks": []}, format="json")
    assert resp.status_code == 503
    data = resp.json()
    assert data["error_code"] == "model_unavailable"
    assert "fallback_models" in data


def test_models_listing(client):
    resp = client.get("/api/reasoning/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "urban_accident_example" in data["models"]
    assert "_template_model" in data["models"]


def test_template_model_infer(client):
    resp = client.post(
        "/api/reasoning/_template_model/infer",
        {"query": "hello", "tracks": []}, format="json")
    assert resp.status_code == 200, resp.content
    assert resp.json()["actions"][0]["label"] == "nominal"
