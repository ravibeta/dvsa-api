"""API tests for the Azure Foundry session-lifecycle endpoints.

These exercise the DRF views end-to-end against a fresh, forced-dry-run session
manager (no DB models are involved, so no ``django_db`` marker is needed).
"""

from __future__ import annotations

import pytest
from rest_framework.test import APIClient

from dvsa_api.reasoning.azure_session_manager import (
    AzureFoundrySessionManager,
    SessionStore,
    reset_session_manager,
)

BASE = "/api/reasoning/foundry"


@pytest.fixture
def client():
    # Point the router's global manager at a clean dry-run instance per test.
    reset_session_manager(
        AzureFoundrySessionManager(store=SessionStore(), dry_run=True))
    yield APIClient()
    reset_session_manager(None)


def _create(client, **body):
    body.setdefault("model_name", "o1-mini")
    return client.post(f"{BASE}/session", body, format="json")


def test_create_session_returns_201_with_contract_fields(client):
    resp = _create(client, max_duration_minutes=15)
    assert resp.status_code == 201, resp.content
    data = resp.json()
    for key in ("session_id", "endpoint", "expires_at", "cost_estimate"):
        assert key in data
    assert data["cost_estimate"] > 0


def test_create_requires_model_name(client):
    resp = client.post(f"{BASE}/session", {}, format="json")
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "invalid_request"


def test_full_lifecycle_create_infer_status_teardown(client):
    sid = _create(client).json()["session_id"]

    # infer
    infer = client.post(
        f"{BASE}/session/{sid}/infer",
        {"query": "accidents?", "tracks": [{"id": "a"}, {"id": "b"}]},
        format="json")
    assert infer.status_code == 200, infer.content
    payload = infer.json()
    assert payload["actions"][0]["label"] == "accident"
    assert payload["metadata"]["session_id"] == sid

    # status reflects one recorded call
    status_resp = client.get(f"{BASE}/session/{sid}")
    assert status_resp.status_code == 200
    assert status_resp.json()["usage"]["calls"] == 1

    # teardown, then status is 404
    td = client.post(f"{BASE}/session/{sid}/teardown", {}, format="json")
    assert td.status_code == 200
    assert td.json()["status"] == "torn_down"

    gone = client.get(f"{BASE}/session/{sid}")
    assert gone.status_code == 404
    assert gone.json()["error_code"] == "session_not_found"


def test_infer_unknown_session_returns_404(client):
    resp = client.post(f"{BASE}/session/nope/infer", {"query": "x"}, format="json")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "session_not_found"


def test_teardown_is_idempotent_over_http(client):
    sid = _create(client).json()["session_id"]
    assert client.post(f"{BASE}/session/{sid}/teardown", {}, format="json").status_code == 200
    # Second teardown still succeeds.
    again = client.post(f"{BASE}/session/{sid}/teardown", {}, format="json")
    assert again.status_code == 200
    assert again.json()["already_gone"] is True
