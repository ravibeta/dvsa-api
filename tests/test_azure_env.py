"""Tests for the session-scoped Azure environment (core.azure).

All tests run in dry-run mode — no Azure SDKs or credentials required — which is
exactly how the package behaves in CI and on a fresh checkout.
"""

import pytest

from core.azure import (
    AzureEnvironmentConfig,
    create_session_azure_environment,
    teardown_session_azure_environment,
)
from core.azure import index_schema, naming


# --------------------------------------------------------------------------
# Index schema
# --------------------------------------------------------------------------
def test_index_schema_has_1536_dim_vector_and_required_fields():
    desc = index_schema.describe_index("idx", dimensions=1536)
    fields = {f["name"]: f for f in desc["fields"]}

    # Required field set from the spec.
    for required in ("vector", "caption", "labels", "tags", "user", "path"):
        assert required in fields, f"missing field {required}"

    assert fields["vector"]["dimensions"] == 1536
    assert fields["vector"]["type"] == "Collection(Edm.Single)"
    # Exactly one key field.
    assert sum(1 for f in desc["fields"] if f.get("key")) == 1


def test_index_schema_respects_custom_dimensions():
    desc = index_schema.describe_index("idx", dimensions=768)
    vector = next(f for f in desc["fields"] if f["name"] == "vector")
    assert vector["dimensions"] == 768


# --------------------------------------------------------------------------
# Naming / isolation
# --------------------------------------------------------------------------
def test_index_isolation_gives_unique_names_per_session():
    a = naming.index_name("dvsa-index", "user-1", "index")
    b = naming.index_name("dvsa-index", "user-2", "index")
    assert a != b
    assert a.startswith("dvsa-index-s-")


def test_filter_isolation_shares_base_index():
    a = naming.index_name("dvsa-index", "user-1", "filter")
    b = naming.index_name("dvsa-index", "user-2", "filter")
    assert a == b == "dvsa-index"


def test_blob_prefix_is_session_scoped():
    assert naming.blob_prefix("User 1!").startswith("sessions/user-1")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def test_config_defaults_to_dryrun_without_subscription():
    cfg = AzureEnvironmentConfig(subscription_id=None, provisioner="auto")
    assert cfg.resolve_mode() == "dryrun"


def test_config_auto_selects_sdk_with_subscription():
    cfg = AzureEnvironmentConfig(subscription_id="sub-123", provisioner="auto")
    assert cfg.resolve_mode() == "sdk"


def test_config_contract_storage_account_name():
    cfg = AzureEnvironmentConfig()
    assert cfg.storage_account == "sadronevideo"
    assert cfg.input_container == "input"
    assert cfg.output_container == "output"
    assert cfg.vector_dimensions == 1536


# --------------------------------------------------------------------------
# Lifecycle (dry-run)
# --------------------------------------------------------------------------
@pytest.fixture
def dryrun_config():
    return AzureEnvironmentConfig(provisioner="dryrun")


def test_create_and_teardown_lifecycle(dryrun_config):
    env = create_session_azure_environment(
        "sess-abc", user_id=7, provision_global=True, config=dryrun_config
    )
    assert env.mode == "dryrun"
    assert env.index_name == "dvsa-index-s-sess-abc"
    assert env.global_provisioned is True

    actions = {(op["action"], op["resource"]) for op in env.operations}
    assert ("ensure", "storage_account") in actions
    assert ("ensure", "containers") in actions
    assert ("ensure", "search_service") in actions
    assert ("ensure", "openai_account") in actions
    assert ("ensure", "deployments") in actions
    assert ("ensure", "search_index") in actions

    result = teardown_session_azure_environment("sess-abc", config=dryrun_config)
    teardown_actions = {(op["action"], op["resource"]) for op in result["operations"]}
    assert ("delete", "search_index") in teardown_actions
    assert ("delete", "blob_prefix") in teardown_actions


def test_storage_account_uses_contract_name(dryrun_config):
    env = create_session_azure_environment(
        "sess-name", provision_global=True, config=dryrun_config
    )
    acct_op = next(o for o in env.operations if o["resource"] == "storage_account")
    assert acct_op["name"] == "sadronevideo"


def test_deployments_include_embedding_and_chat(dryrun_config):
    env = create_session_azure_environment(
        "sess-dep", provision_global=True, config=dryrun_config
    )
    dep_op = next(o for o in env.operations if o["resource"] == "deployments")
    roles = {d["role"] for d in dep_op["deployments"]}
    assert {"embedding", "chat"} <= roles


def test_process_frame_dryrun_produces_1536_vector(dryrun_config):
    env = create_session_azure_environment("sess-proc", user_id=3, config=dryrun_config)
    out = env.process_frame(
        frame_id="f001", description="parking lot with 4 cars",
        path="f001.jpg",
    )
    assert out["frame_id"] == "f001"
    assert out["path"].startswith("sessions/sess-proc/")
    assert out["dryrun"] is True
    # Embedding width matches the index contract.
    assert len(env.embed("anything")) == 1536


def test_embed_is_deterministic(dryrun_config):
    env = create_session_azure_environment("sess-embed", config=dryrun_config)
    assert env.embed("same text") == env.embed("same text")
    assert env.embed("a") != env.embed("b")
