"""Agents-as-code: yaml loading, generated function tools, idempotent sync."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.azure import foundry_provisioning as fp

AGENTS_DIR = Path(__file__).resolve().parents[1] / "solution-accelerator" / "agents"
ENV = {"SUBSCRIPTION_ID": "sub-1", "RESOURCE_GROUP": "rg-1", "ACCOUNT": "acct", "PROJECT": "proj"}


class FakeClient:
    """In-memory stand-in for FoundryProjectClient."""

    def __init__(self, existing=None, fail=()):
        self.agents = dict(existing or {})
        self.created = []
        self.fail = set(fail)

    def get_agent(self, name):
        return self.agents.get(name)

    def create_agent_version(self, spec):
        if spec.name in self.fail:
            raise fp.FoundryProvisioningError("boom")
        self.created.append(spec.name)
        v = str(len(self.created))
        self.agents[spec.name] = {"versions": {"latest": {"version": v, "definition": spec.definition}}}
        return {"version": v}


def _spec(name="a", **defn):
    base = {"kind": "prompt", "model": "gpt-4o-mini", "instructions": "x", "tools": []}
    return fp.AgentSpec(name=name, definition={**base, **defn})


# --- yaml loading ----------------------------------------------------------
def test_loads_every_agent_yaml_and_includes_qwen():
    specs = {s.name: s for s in fp.load_agent_specs(AGENTS_DIR, ENV)}
    assert {"tool-agent-in-a-team", "fn-agent-in-a-team", "chat-agent-in-a-team",
            "master-agent-in-a-team", "scene-search-agent", "qwen-agent-in-a-team"} <= set(specs)
    q = specs["qwen-agent-in-a-team"].definition
    assert q["model"] == "qwen--qwen35-08b-2" and q["tools"] == []


def test_runtime_artifacts_and_hardcoded_ids_are_not_sent():
    for s in fp.load_agent_specs(AGENTS_DIR, ENV):
        blob = str(s.definition) + str(s.metadata)
        assert "656e67c6" not in blob, s.name           # no hardcoded subscription
        assert "${" not in blob, s.name                  # all placeholders resolved
        assert "instance_identity" not in blob and "blueprint" not in blob


def test_connection_ids_are_templated():
    master = next(s for s in fp.load_agent_specs(AGENTS_DIR, ENV) if s.name == "master-agent-in-a-team")
    conn = master.definition["tools"][1]["azure_ai_search"]["indexes"][0]["project_connection_id"]
    assert conn.startswith("/subscriptions/sub-1/resourceGroups/rg-1/")
    assert conn.endswith("/accounts/acct/projects/proj/connections/srch-vision-01")


def test_env_from_project_endpoint():
    env = fp.env_from_project_endpoint(
        "https://found-vision-1.services.ai.azure.com/api/projects/droneimage",
        subscription_id="s", resource_group="r")
    assert env == {"SUBSCRIPTION_ID": "s", "RESOURCE_GROUP": "r",
                   "ACCOUNT": "found-vision-1", "PROJECT": "droneimage"}


# --- generated function tools ---------------------------------------------
def test_tool_agent_function_tools_come_from_analyzer_functions():
    from core.azure.analyzer import analyzer_functions

    spec = fp.apply_function_tools(
        next(s for s in fp.load_agent_specs(AGENTS_DIR, ENV) if s.name == "tool-agent-in-a-team"))
    fns = {t["name"]: t for t in spec.definition["tools"] if t["type"] == "function"}
    assert set(fns) == {f.__name__ for f in analyzer_functions()}
    assert "describe_frame" in fns and "plot_clusters" not in fns  # stale names gone
    assert fns["describe_frame"]["parameters"]["type"] == "object"
    assert "account_id" in fns["describe_frame"]["parameters"]["properties"]
    # non-function tools (web_search, KB MCP servers) are preserved
    assert any(t["type"] == "mcp" for t in spec.definition["tools"])
    assert any(t["type"] == "web_search" for t in spec.definition["tools"])


def test_agents_without_a_source_are_untouched():
    s = _spec("chat-agent-in-a-team", tools=[{"type": "web_search"}])
    assert fp.apply_function_tools(s).definition["tools"] == [{"type": "web_search"}]


# --- idempotent sync -------------------------------------------------------
def test_sync_creates_missing_agent():
    c = FakeClient()
    res = fp.sync_agents(c, [_spec("new")])
    assert res == [{"agent": "new", "action": "create", "version": "1", "tools": 0}]


def test_sync_is_idempotent_when_definition_matches():
    c = FakeClient()
    fp.sync_agents(c, [_spec("a")])
    res = fp.sync_agents(c, [_spec("a")])
    assert res[0]["action"] == "unchanged" and c.created == ["a"]


def test_sync_creates_new_version_when_definition_differs_and_ignores_tool_order():
    t1, t2 = {"type": "web_search"}, {"type": "function", "name": "f"}
    c = FakeClient({"a": {"versions": {"latest": {"version": "1", "definition": {
        "kind": "prompt", "model": "gpt-4o-mini", "instructions": "x", "tools": [t1, t2]}}}}})
    assert fp.sync_agents(c, [_spec("a", tools=[t2, t1])])[0]["action"] == "unchanged"
    assert fp.sync_agents(c, [_spec("a", instructions="changed", tools=[t2, t1])])[0]["action"] == "update"


def test_dry_run_changes_nothing_and_only_filter_applies():
    c = FakeClient()
    res = fp.sync_agents(c, [_spec("a"), _spec("b")], dry_run=True, only=["b"])
    assert [r["agent"] for r in res] == ["b"] and res[0]["action"] == "would-create"
    assert c.created == []


def test_one_agent_failing_does_not_stop_the_rest():
    c = FakeClient(fail={"bad"})
    res = {r["agent"]: r for r in fp.sync_agents(c, [_spec("bad"), _spec("good")])}
    assert res["bad"]["action"] == "error" and res["good"]["action"] == "create"


# --- REST client -----------------------------------------------------------
class Resp:
    def __init__(self, status, payload=None):
        self.status_code, self._p, self.text = status, payload or {}, str(payload)
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._p


class Sess:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self.responses.pop(0)


def _client(sess):
    return fp.FoundryProjectClient(
        "https://x.services.ai.azure.com/api/projects/p/", session=sess, retries=2,
        retry_wait=0, credential=SimpleNamespace(get_token=lambda s: SimpleNamespace(token="t")))


def test_client_retries_rbac_propagation_then_succeeds(monkeypatch):
    monkeypatch.setattr(fp.time, "sleep", lambda *_: None)
    sess = Sess(Resp(403, {}), Resp(200, {"name": "a"}))
    assert _client(sess).get_agent("a") == {"name": "a"}
    assert len(sess.calls) == 2
    method, url, kw = sess.calls[0]
    assert url == "https://x.services.ai.azure.com/api/projects/p/agents/a?api-version=v1"
    assert kw["headers"]["Authorization"] == "Bearer t"


def test_client_404_is_none_and_hard_errors_raise(monkeypatch):
    monkeypatch.setattr(fp.time, "sleep", lambda *_: None)
    assert _client(Sess(Resp(404, {}))).get_agent("nope") is None
    with pytest.raises(fp.FoundryProvisioningError):
        _client(Sess(Resp(500, {"e": 1}))).get_agent("a")


def test_client_posts_definition_as_new_version():
    sess = Sess(Resp(200, {"version": "4"}))
    out = _client(sess).create_agent_version(_spec("a"))
    method, url, kw = sess.calls[0]
    assert method == "POST" and url.endswith("/agents/a/versions?api-version=v1")
    assert out == {"version": "4"} and '"definition"' in kw["data"]
