"""Unit tests for AzureSdkProvisioner.ensure_search_index's non-fatal fallback.

Live bug: ensure_search_index runs on *every* session creation (every chat/
extract-frames/ingest request), unconditionally re-asserting the index's full
schema via create_or_update_index even though the index almost always already
exists unchanged. Once a Knowledge Agent (run_connected_agent) targeted the
index directly, Azure AI Search started auto-managing a backing knowledge
source tied to the index's semantic configuration; our schema doesn't declare
one, so Azure rejected the update as "removing" it — and that raised
ProvisioningError, which 500'd the *entire* chat request even though the
index itself was perfectly usable (we'd ingested into it successfully
earlier the same session). A redundant "ensure" step must never take down an
otherwise-working request.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from core.azure.provisioning import AzureSdkProvisioner


def _provisioner():
    config = SimpleNamespace(
        subscription_id="sub-1", resource_group="rg-1",
        search_endpoint="https://search.example.net", search_admin_key="key",
        vector_dimensions=1536, account_key=None,
    )
    config.control_plane_ready = lambda: True
    return AzureSdkProvisioner(config)


def test_ensure_search_index_degrades_gracefully_on_semantic_config_conflict(monkeypatch):
    provisioner = _provisioner()
    fake_index_client = MagicMock()
    fake_index_client.create_or_update_index.side_effect = Exception(
        "(OperationNotAllowed) You cannot remove the semantic configuration for "
        "this index because knowledge source(s) 'knowledgesource-123' are still "
        "referencing it."
    )
    monkeypatch.setattr(provisioner, "_index_client", lambda: fake_index_client)

    # Must not raise — this is what previously propagated up through
    # session.py -> views.py and 500'd the whole chat request.
    result = provisioner.ensure_search_index("dvsa-index-s-account-5")

    assert result["name"] == "dvsa-index-s-account-5"
    assert "error" in result


def test_ensure_search_index_succeeds_normally_when_no_conflict(monkeypatch):
    provisioner = _provisioner()
    fake_index_client = MagicMock()
    monkeypatch.setattr(provisioner, "_index_client", lambda: fake_index_client)

    result = provisioner.ensure_search_index("dvsa-index-s-account-5")

    fake_index_client.create_or_update_index.assert_called_once()
    assert result["name"] == "dvsa-index-s-account-5"
    assert "error" not in result
