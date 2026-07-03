"""Tests for the Azure Foundry reasoning adapter.

Covers both managed-session and direct-endpoint modes, response mapping and
telemetry — all offline, with the network call short-circuited by synthetic
endpoints or monkeypatching.
"""

from __future__ import annotations

import json
import os

import pytest

from dvsa_api.reasoning.adapter_base import ensure_response_shape
from dvsa_api.reasoning.azure_foundry_adapter import (
    AzureFoundryAdapter,
    _map_foundry_response,
    _synthetic_reasoning,
)
from dvsa_api.reasoning.azure_session_manager import (
    AzureFoundrySessionManager,
    SessionStore,
)
from dvsa_api.reasoning.errors import ModelUnavailableError

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def manager():
    return AzureFoundrySessionManager(store=SessionStore(), dry_run=True)


# --------------------------------------------------------------------------
# Managed-session predict (synthetic, offline)
# --------------------------------------------------------------------------
def test_managed_predict_flags_accident_with_two_tracks(manager):
    session = manager.create_session(model_name="o1-mini")
    adapter = AzureFoundryAdapter(session_id=session.session_id, manager=manager)
    context = {"query": "accidents?", "tracks": [{"id": "a"}, {"id": "b"}]}
    resp = adapter.predict(context)

    assert set(("actions", "reasoning_trace", "metadata")).issubset(resp)
    assert resp["actions"][0]["label"] == "accident"
    meta = resp["metadata"]
    # Telemetry contract.
    assert meta["model"] == "o1-mini"
    assert meta["session_id"] == session.session_id
    assert meta["mode"] == "managed"
    assert meta["provider"] == "azure_foundry"
    assert isinstance(meta["latency_ms"], int)
    assert "cost_estimate_ms" in meta
    assert meta["foundry_reasoning_effort"] == "medium"

    # Usage was accrued back onto the session.
    assert manager.get_session(session.session_id).usage.calls == 1


def test_managed_predict_nominal_with_few_tracks(manager):
    session = manager.create_session(model_name="o1-mini")
    adapter = AzureFoundryAdapter(session_id=session.session_id, manager=manager)
    resp = adapter.predict({"query": "scene?", "tracks": [{"id": "a"}]})
    assert resp["actions"][0]["label"] == "nominal"


def test_predict_unknown_session_raises(manager):
    adapter = AzureFoundryAdapter(session_id="nope", manager=manager)
    with pytest.raises(Exception):  # SessionNotFoundError (a ReasoningError)
        adapter.predict({"query": "x"})


# --------------------------------------------------------------------------
# Direct-endpoint mode
# --------------------------------------------------------------------------
def test_direct_mode_synthetic_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT",
                       "https://foundry-x.example-foundry.local/score")
    adapter = AzureFoundryAdapter(model_name="o1-mini")
    assert adapter.is_direct
    resp = adapter.predict({"query": "q", "tracks": [{"id": "a"}, {"id": "b"}]})
    assert resp["metadata"]["mode"] == "direct"
    assert resp["actions"][0]["label"] == "accident"


def test_direct_mode_real_endpoint_calls_network(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://real.example.com/score")
    monkeypatch.setenv("AZURE_FOUNDRY_KEY", "k")
    adapter = AzureFoundryAdapter(model_name="o1-mini")
    # Intercept the HTTP layer so no real request is made.
    native = _load("foundry_response_native.json")
    monkeypatch.setattr(adapter, "_call_foundry",
                        lambda *a, **k: _map_foundry_response(native, {}, "o1-mini"))
    resp = adapter.predict({"query": "q"})
    assert resp["actions"][0]["label"] == "accident"
    assert resp["metadata"]["tokens"] == 128


# --------------------------------------------------------------------------
# Response mapping
# --------------------------------------------------------------------------
def test_map_native_response():
    mapped = _map_foundry_response(_load("foundry_response_native.json"), {}, "o1-mini")
    assert mapped["actions"][0]["confidence"] == 0.91
    assert mapped["tokens"] == 128
    assert mapped["reasoning_effort"] == "high"


def test_map_openai_style_response():
    mapped = _map_foundry_response(_load("foundry_response_openai.json"), {}, "o1-mini")
    assert mapped["actions"][0]["label"] == "nominal"
    assert mapped["tokens"] == 64


def test_synthetic_reasoning_is_deterministic():
    ctx = {"query": "q", "tracks": [{"id": "a"}, {"id": "b"}]}
    assert _synthetic_reasoning(ctx, "o1-mini") == _synthetic_reasoning(ctx, "o1-mini")


# --------------------------------------------------------------------------
# health_check + response shape validation
# --------------------------------------------------------------------------
def test_health_check_managed_ok(manager):
    session = manager.create_session(model_name="o1-mini")
    adapter = AzureFoundryAdapter(session_id=session.session_id, manager=manager)
    hc = adapter.health_check()
    assert hc["status"] == "ok" and hc["mode"] == "managed"


def test_health_check_not_configured(manager, monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_FOUNDY_ENDPOINT", raising=False)
    adapter = AzureFoundryAdapter(manager=manager)
    assert adapter.health_check()["status"] == "error"


def test_ensure_response_shape_rejects_missing_metadata():
    with pytest.raises(ValueError):
        ensure_response_shape({"actions": [], "reasoning_trace": []})


def test_adapter_without_target_raises(manager, monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_FOUNDY_ENDPOINT", raising=False)
    adapter = AzureFoundryAdapter(manager=manager)
    with pytest.raises(ModelUnavailableError):
        adapter.predict({"query": "x"})
