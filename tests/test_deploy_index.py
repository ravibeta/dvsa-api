"""core.azure.deploy.ensure_base_index: update in place, else safely recreate."""

from types import SimpleNamespace

import pytest

import core.azure.provisioning as provisioning
from core.azure import deploy

CFG = SimpleNamespace(search_index_name="dvsa-index", search_endpoint="https://s", search_admin_key="k")


class FakeProv:
    """AzureSdkProvisioner stand-in: first ensure fails (schema conflict) unless told otherwise."""

    instances = []

    def __init__(self, config, fail_first=True, docs=0, blocked=False):
        self.calls, self.fail_first, self.docs, self.dropped = [], fail_first, docs, False
        self.blocked = blocked
        FakeProv.instances.append(self)

    def ensure_search_index(self, name):
        self.calls.append("ensure")
        if self.fail_first and not self.dropped:
            return {"name": name, "error": "(OperationNotAllowed) cannot change field type"}
        return {"name": name}

    def _index_client(self):
        def delete_index(name):
            self.calls.append("drop")
            if self.blocked:
                raise RuntimeError("Cannot delete index 'x' because agent(s) 'search-agent-in-a-team' "
                                   "are still referencing it.")
            self.dropped = True

        docs = self.docs
        return SimpleNamespace(delete_index=delete_index,
                               get_search_client=lambda n: SimpleNamespace(get_document_count=lambda: docs))


@pytest.fixture
def prov(monkeypatch):
    def install(**kw):
        monkeypatch.setattr(provisioning, "AzureSdkProvisioner", lambda c: FakeProv(c, **kw))
        FakeProv.instances.clear()
        return FakeProv.instances
    return install


def test_in_place_update_when_schema_is_compatible(prov):
    inst = prov(fail_first=False)
    assert deploy.ensure_base_index(CFG) == {"index": "dvsa-index", "action": "ensured"}
    assert inst[0].calls == ["ensure"]


def test_empty_legacy_index_is_dropped_and_rebuilt(prov):
    inst = prov(docs=0)
    out = deploy.ensure_base_index(CFG)
    assert out["action"] == "recreated" and out["dropped_documents"] == 0
    assert inst[0].calls == ["ensure", "drop", "ensure"]


def test_index_with_documents_is_never_dropped_without_the_flag(prov):
    inst = prov(docs=42)
    out = deploy.ensure_base_index(CFG)
    assert out["action"] == "error" and out["documents"] == 42 and "--recreate-index" in out["error"]
    assert "drop" not in inst[0].calls


def test_recreate_flag_forces_rebuild_of_a_populated_index(prov):
    inst = prov(docs=42)
    out = deploy.ensure_base_index(CFG, recreate=True)
    assert out["action"] == "recreated" and out["dropped_documents"] == 42
    assert inst[0].calls == ["ensure", "drop", "ensure"]


def test_dry_run_touches_nothing(prov):
    inst = prov()
    cfg = SimpleNamespace(search_index_name="dvsa-index", search_endpoint="https://s", search_admin_key="k",
                          vector_dimensions=1536, openai_endpoint=None,
                          embedding_deployment="e", embedding_model="m")
    out = deploy.ensure_base_index(cfg, dry_run=True)
    assert out["action"] == "would-ensure" and out["semantic"] == "dvsa-semantic"
    assert inst == []


def test_index_referenced_by_a_knowledge_agent_is_reported_blocked_not_dropped(prov):
    inst = prov(docs=0, blocked=True)
    out = deploy.ensure_base_index(CFG)
    assert out["action"] == "blocked" and "knowledge agent" in out["error"]
    assert "search-agent-in-a-team" in out["detail"] and out["remediation"]
    assert inst[0].calls == ["ensure", "drop"]          # tried once, never re-ensured


def test_blocked_index_is_a_warning_not_a_failed_deploy(monkeypatch):
    from core.azure import config as cfgmod

    class C(SimpleNamespace):
        pass

    monkeypatch.setattr(cfgmod.AzureEnvironmentConfig, "from_settings",
                        classmethod(lambda cls: C(search_index_name="dvsa-index", project_endpoint=None)))
    monkeypatch.setattr(deploy, "ensure_base_index",
                        lambda *a, **k: {"index": "dvsa-index", "action": "blocked"})
    args = SimpleNamespace(skip_index=False, skip_agents=True, index=None, dry_run=False, recreate_index=False)
    assert deploy.run(args)["ok"] is True

    monkeypatch.setattr(deploy, "ensure_base_index",
                        lambda *a, **k: {"index": "dvsa-index", "action": "error", "error": "x"})
    assert deploy.run(args)["ok"] is False
