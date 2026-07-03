"""Tests for the Azure Foundry session manager.

All tests run offline (dry-run): no Terraform binary, no Azure SDK, no network.
The SDK provisioning path is validated with an in-memory fake wrapper.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

from dvsa_api.reasoning import azure_session_manager as asm
from dvsa_api.reasoning.azure_session_manager import (
    AzureFoundrySessionManager,
    SessionStore,
    estimate_session_cost,
)
from dvsa_api.reasoning.errors import (
    CostLimitExceeded,
    QuotaExceeded,
    SessionNotFoundError,
)


@pytest.fixture
def manager():
    """A fresh, forced dry-run manager with an isolated store per test."""
    return AzureFoundrySessionManager(store=SessionStore(), dry_run=True)


# --------------------------------------------------------------------------
# Cost estimation + creation
# --------------------------------------------------------------------------
def test_estimate_session_cost_scales_with_duration():
    hourly = asm.sku_hourly_rate("o1-mini")
    assert estimate_session_cost("o1-mini", 60) == pytest.approx(hourly)
    assert estimate_session_cost("o1-mini", 30) == pytest.approx(hourly / 2)


def test_create_session_dryrun_returns_synthetic_endpoint(manager):
    session = manager.create_session(model_name="o1-mini", max_duration_minutes=30)
    assert session.provisioner == "dryrun"
    assert session.status == "active"
    assert session.endpoint_url.endswith("/score")
    assert ".example-foundry.local" in session.endpoint_url
    assert session.cost_estimate_usd == estimate_session_cost("o1-mini", 30)
    # Secret was written to the (in-memory) vault during provisioning.
    assert manager.sdk.get_secret(session.key_vault_secret_name).startswith("dryrun-key")
    # Public view hides the secret name and serializes timestamps.
    pub = session.to_public_dict()
    assert "key_vault_secret_name" not in pub
    assert isinstance(pub["expires_at"], str)


def test_create_rejects_when_estimate_exceeds_cost_cap(manager):
    # o1 is pricey; 60 min at a 0.10 cap must be rejected up-front.
    with pytest.raises(CostLimitExceeded):
        manager.create_session(model_name="o1", max_duration_minutes=60,
                               max_cost_usd=0.10)


# --------------------------------------------------------------------------
# Quotas
# --------------------------------------------------------------------------
def test_global_quota_enforced(manager, monkeypatch):
    monkeypatch.setattr(asm, "MAX_ACTIVE_SESSIONS", 2)
    manager.create_session(model_name="o1-mini")
    manager.create_session(model_name="o1-mini")
    with pytest.raises(QuotaExceeded):
        manager.create_session(model_name="o1-mini")


def test_per_user_quota_enforced(manager, monkeypatch):
    monkeypatch.setattr(asm, "MAX_ACTIVE_SESSIONS_PER_USER", 1)
    manager.create_session(model_name="o1-mini", user_id="u1")
    with pytest.raises(QuotaExceeded):
        manager.create_session(model_name="o1-mini", user_id="u1")
    # Different user is unaffected.
    manager.create_session(model_name="o1-mini", user_id="u2")


# --------------------------------------------------------------------------
# Usage accrual + cost stops
# --------------------------------------------------------------------------
def test_record_usage_accrues_and_hard_stops(manager):
    session = manager.create_session(model_name="o1-mini", max_cost_usd=0.5)
    sid = session.session_id
    # A huge token count drives the accrued cost past the hard cap.
    updated = manager.record_usage(sid, tokens=1_000_000, latency_ms=10)
    assert updated.status == "over_budget"
    # Hard stop tore the session down, so it is gone from the store.
    with pytest.raises(SessionNotFoundError):
        manager.get_session(sid)


def test_soft_stop_blocks_infer_until_forced(manager):
    session = manager.create_session(model_name="o1-mini", max_cost_usd=1.0)
    sid = session.session_id
    # Nudge accrued cost just past the soft limit (80% of cap) but under hard.
    session.usage.estimated_cost_usd = session.soft_limit_usd() + 0.001
    manager.store.put(session)
    with pytest.raises(CostLimitExceeded):
        manager.check_can_infer(sid)
    # force=True bypasses the soft stop.
    assert manager.check_can_infer(sid, force=True).session_id == sid


# --------------------------------------------------------------------------
# Teardown idempotency + sweeping
# --------------------------------------------------------------------------
def test_teardown_is_idempotent(manager):
    session = manager.create_session(model_name="o1-mini")
    sid = session.session_id
    first = manager.teardown_session(sid)
    assert first["status"] == "torn_down"
    # Repeated teardown (and teardown of unknown id) must succeed, not raise.
    second = manager.teardown_session(sid)
    assert second["already_gone"] is True
    assert manager.teardown_session("does-not-exist")["status"] == "torn_down"


def test_sweep_reaps_expired_and_inactive(manager):
    expired = manager.create_session(model_name="o1-mini")
    expired.expires_at = expired.created_at - timedelta(minutes=1)
    manager.store.put(expired)

    inactive = manager.create_session(model_name="o1-mini")
    inactive.last_activity_at = inactive.created_at - timedelta(hours=2)
    manager.store.put(inactive)

    fresh = manager.create_session(model_name="o1-mini")

    reaped = manager.sweep()
    reaped_ids = {r["session_id"] for r in reaped}
    assert expired.session_id in reaped_ids
    assert inactive.session_id in reaped_ids
    assert fresh.session_id not in reaped_ids
    assert manager.get_session(fresh.session_id).session_id == fresh.session_id


# --------------------------------------------------------------------------
# SDK provisioning path (fake wrapper — still no network)
# --------------------------------------------------------------------------
class _FakeSdk:
    def __init__(self):
        self.secrets = {}
        self.rgs = {}

    def ensure_resource_group(self, name, location):
        self.rgs[name] = location
        return {"name": name, "location": location, "provisioned": True}

    def delete_resource_group(self, name):
        self.rgs.pop(name, None)
        return {"name": name, "deleted": True}

    def set_secret(self, name, value):
        self.secrets[name] = value
        return name

    def get_secret(self, name):
        return self.secrets.get(name, "missing")

    def delete_secret(self, name):
        self.secrets.pop(name, None)


def test_sdk_provisioning_uses_wrapper():
    fake = _FakeSdk()
    mgr = AzureFoundrySessionManager(
        store=SessionStore(), sdk=fake, use_sdk_provision=True, dry_run=False)
    session = mgr.create_session(model_name="o1-mini")
    assert session.provisioner == "sdk"
    assert session.resource_group in fake.rgs
    assert session.key_vault_secret_name in fake.secrets
    # Teardown deletes the resource group and scrubs the secret.
    mgr.teardown_session(session.session_id)
    assert session.resource_group not in fake.rgs
    assert session.key_vault_secret_name not in fake.secrets


# --------------------------------------------------------------------------
# Terraform module presence (+ validate when the binary is available)
# --------------------------------------------------------------------------
def test_terraform_module_files_exist():
    base = os.path.join("infra", "azure", "foundry_module")
    for name in ("versions.tf", "variables.tf", "main.tf", "outputs.tf", "README.md"):
        path = os.path.join(base, name)
        assert os.path.isfile(path), f"missing {path}"
        assert os.path.getsize(path) > 0
