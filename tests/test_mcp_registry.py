"""Discovery + registry + capability-selection tests for the MCP runtime.

Pure-Python: no Django, no network. Exercises folder discovery of ``mcp/agents/*``,
lazy agent loading + contract validation, the allow-list, and ``select_agent``
policies.
"""

import pytest

from dvsa_api.mcp import registry
from dvsa_api.mcp.errors import AgentNotAllowedError, AgentUnavailableError


@pytest.fixture()
def discovered(monkeypatch):
    monkeypatch.delenv("MCP_ALLOWED_AGENTS", raising=False)
    monkeypatch.delenv("MCP_AGENT_WHITELIST", raising=False)
    registry.discover_agents(force=True)
    yield registry


# --------------------------------------------------------------------------- #
# Discovery + loading
# --------------------------------------------------------------------------- #
def test_discovery_finds_template_and_examples(discovered):
    names = discovered.list_agents()
    for expected in ("_template_agent", "scout_agent", "analyzer_agent",
                     "coordinator_agent", "human_interface_agent"):
        assert expected in names


def test_get_agent_returns_healthy_adapter(discovered):
    for name in ("scout_agent", "analyzer_agent", "coordinator_agent"):
        agent = discovered.get_agent(name)
        assert callable(agent.handle_task)
        assert agent.health_check()["status"] == "ok"


def test_manifest_capabilities_parsed(discovered):
    manifest = discovered.get_manifest("scout_agent")
    assert manifest is not None
    assert "scout" in manifest.capabilities
    assert manifest.type == "inprocess"


def test_unknown_agent_raises(discovered):
    with pytest.raises(AgentUnavailableError):
        discovered.get_agent("does_not_exist")


# --------------------------------------------------------------------------- #
# Allow-list
# --------------------------------------------------------------------------- #
def test_allowlist_blocks_unlisted_agent(discovered, monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_AGENTS", "scout_agent")
    with pytest.raises(AgentNotAllowedError):
        discovered.get_agent("analyzer_agent")
    # The allowed agent still loads.
    assert discovered.get_agent("scout_agent").health_check()["status"] == "ok"


# --------------------------------------------------------------------------- #
# Capability selection policies
# --------------------------------------------------------------------------- #
def test_select_agent_by_capability(discovered):
    assert discovered.select_agent("scout", "priority") == "scout_agent"
    assert discovered.select_agent("analyzer", "latency_optimized") == "analyzer_agent"


def test_select_agent_round_robin_cycles(discovered):
    # Only one agent per capability here, so round robin returns it stably.
    a = discovered.select_agent("coordinator", "round_robin")
    b = discovered.select_agent("coordinator", "round_robin")
    assert a == b == "coordinator_agent"


def test_select_unknown_capability_raises(discovered):
    with pytest.raises(AgentUnavailableError):
        discovered.select_agent("no_such_capability", "priority")


def test_select_unknown_policy_raises(discovered):
    with pytest.raises(AgentUnavailableError):
        discovered.select_agent("scout", "magic_policy")
