"""Unit tests for threading account_id/video_id context into the chat agent.

``ChatAPIView`` only ever gave the function-tool agent the raw question text,
with no account_id/video_id, so the model had no way to call tools like
``get_sas_url_template``/``ask_perplexity`` with the right IDs and would just
ask the user for an image instead of resolving one itself. ``_run_function_agent``
now prefixes the thread content with a ``[Context: account_id=..., video_id=...]``
line whenever an account_id is given, and keeps a previously-created agent's
instructions in sync so it also understands that context.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from core.azure.agents import FoundryAgents


def _agents_obj():
    config = SimpleNamespace(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p",
        agent_model="gpt-4.1-mini",
        tool_agent_name="tool-agent-in-a-team",
        fn_agent_name="fn-agent-in-a-team",
    )
    return FoundryAgents(config)


def test_context_prefix_injected_for_new_agent(monkeypatch):
    agents = _agents_obj()
    fake_client = MagicMock()
    new_agent = SimpleNamespace(id="agent-1", instructions="whatever")
    fake_client.create_agent.return_value = new_agent
    captured = {}

    def fake_run_agent(agents_client, agent, content, tool_executor):
        captured["content"] = content
        return "ok"

    monkeypatch.setattr(agents, "_agents_client", lambda: fake_client)
    monkeypatch.setattr(agents, "_find_agent", lambda client, name: None)
    monkeypatch.setattr(agents, "_run_agent", fake_run_agent)

    result = agents.run_analyzer_tools("Describe the scene", "5", video_id="2")

    assert result == "ok"
    assert captured["content"] == (
        "[Context: account_id=5, video_id=2]\nDescribe the scene"
    )
    fake_client.create_agent.assert_called_once()


def test_context_prefix_omitted_without_account_id(monkeypatch):
    agents = _agents_obj()
    fake_client = MagicMock()
    fake_client.create_agent.return_value = SimpleNamespace(id="agent-1", instructions="x")
    captured = {}

    def fake_run_agent(agents_client, agent, content, tool_executor):
        captured["content"] = content
        return "ok"

    monkeypatch.setattr(agents, "_agents_client", lambda: fake_client)
    monkeypatch.setattr(agents, "_find_agent", lambda client, name: None)
    monkeypatch.setattr(agents, "_run_agent", fake_run_agent)

    agents._run_function_agent("raw question", "some-agent", set())

    assert captured["content"] == "raw question"


def test_stale_instructions_are_synced_on_existing_agent(monkeypatch):
    agents = _agents_obj()
    fake_client = MagicMock()
    existing_agent = SimpleNamespace(id="agent-1", instructions="stale instructions")
    updated_agent = SimpleNamespace(id="agent-1", instructions="new instructions")
    fake_client.update_agent.return_value = updated_agent

    monkeypatch.setattr(agents, "_agents_client", lambda: fake_client)
    monkeypatch.setattr(agents, "_find_agent", lambda client, name: existing_agent)
    monkeypatch.setattr(agents, "_run_agent", lambda *a, **k: "ok")

    agents.run_analyzer_tools("Describe the scene", "5", video_id="2")

    fake_client.update_agent.assert_called_once()
    assert fake_client.update_agent.call_args.args[0] == "agent-1"
    fake_client.create_agent.assert_not_called()


def test_chat_api_view_passes_video_id_through(monkeypatch):
    """ChatAPIView previously read video_id from the request but never used it."""
    import apps.videos.views as views

    captured = {}

    class FakeEnv:
        def ask(self, query_text, account_id, video_id=None):
            captured["query_text"] = query_text
            captured["account_id"] = account_id
            captured["video_id"] = video_id
            return "a grounded answer"

    monkeypatch.setattr(views, "create_session_azure_environment", lambda *a, **k: FakeEnv())

    view = views.ChatAPIView()
    request = SimpleNamespace(
        data={"account_id": "5", "query": "Describe the scene", "video_id": 2},
        user=SimpleNamespace(pk=1),
    )
    response = view.put(request)

    assert response.status_code == 200
    assert captured == {
        "query_text": "Describe the scene", "account_id": "5", "video_id": "2",
    }
