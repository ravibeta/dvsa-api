"""Discovery + registry + policy-routing tests for the pluggable reasoning layer.

Pure-Python: no Django, no network, no model binaries. Exercises folder discovery
of ``models/reasoning/*``, lazy adapter loading, the contract validation, and the
``select_model`` policy engine.
"""

import pytest

from dvsa_api.reasoning import registry


@pytest.fixture()
def discovered():
    """Ensure the on-disk model folders are discovered (idempotent)."""
    registry.discover_models(force=True)
    yield registry


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_discovery_finds_template_and_example(discovered):
    names = discovered.list_models()
    assert "_template_model" in names
    assert "urban_accident_example" in names


def test_manifests_parsed(discovered):
    manifest = discovered.get_manifest("urban_accident_example")
    assert manifest is not None
    assert manifest.type == "local"
    assert manifest.version == "1.0.0"
    assert "anomaly_detection" in manifest.capabilities


# --------------------------------------------------------------------------- #
# Adapter loading + contract
# --------------------------------------------------------------------------- #
def test_get_model_returns_adapter_with_ok_health(discovered):
    for name in ("_template_model", "urban_accident_example"):
        adapter = discovered.get_model(name)
        assert callable(getattr(adapter, "predict"))
        assert adapter.health_check()["status"] == "ok"


def test_get_model_caches_instance(discovered):
    a = discovered.get_model("urban_accident_example")
    b = discovered.get_model("urban_accident_example")
    assert a is b


def test_unknown_model_raises(discovered):
    from dvsa_api.reasoning.errors import ModelUnavailableError
    with pytest.raises(ModelUnavailableError):
        discovered.get_model("does_not_exist")


def test_call_model_runs_predict(discovered):
    out = discovered.call_model("urban_accident_example", {"tracks": []})
    assert set(out) >= {"actions", "reasoning_trace", "metadata"}


# --------------------------------------------------------------------------- #
# Policy engine
# --------------------------------------------------------------------------- #
def test_select_latency_optimized_prefers_lowest_hint(discovered):
    # urban_accident_example has latency_hint_ms=30 vs template's 40.
    assert discovered.select_model("latency_optimized", {}) == "urban_accident_example"


def test_select_privacy_first_stays_local(discovered):
    chosen = discovered.select_model("privacy_first", {})
    assert discovered.get_manifest(chosen).is_local


def test_select_by_name_uses_context(discovered):
    assert discovered.select_model(
        "by_name", {"model": "_template_model"}) == "_template_model"


def test_select_by_name_missing_raises(discovered):
    from dvsa_api.reasoning.errors import ModelUnavailableError
    with pytest.raises(ModelUnavailableError):
        discovered.select_model("by_name", {"model": "nope"})


def test_unknown_policy_raises(discovered):
    from dvsa_api.reasoning.errors import ModelUnavailableError
    with pytest.raises(ModelUnavailableError):
        discovered.select_model("magic", {})
