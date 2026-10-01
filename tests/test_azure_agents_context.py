"""Unit tests for threading account_id/video_id context into the chat agent.

``ChatAPIView`` only ever gave the function-tool agent the raw question text,
with no account_id/video_id, so the model had no way to call tools like
``get_sas_url_template``/``ask_perplexity`` with the right IDs and would just
ask the user for an image instead of resolving one itself. ``_run_function_agent``
now prefixes the thread content with a ``[Context: account_id=..., video_id=...]``
line (plus the video's curated extracted-frame URLs, when any exist) whenever
an account_id is given, and keeps a previously-created agent's instructions in
sync so it also understands that context.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.azure.agents import FoundryAgents, _account_video_context


def _agents_obj():
    config = SimpleNamespace(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p",
        agent_model="gpt-4.1-mini",
        tool_agent_name="tool-agent-in-a-team",
        fn_agent_name="fn-agent-in-a-team",
    )
    return FoundryAgents(config)


def test_context_prefix_injected_without_frames(monkeypatch):
    import core.azure.analyzer as analyzer_module

    agents = _agents_obj()
    fake_client = MagicMock()
    new_agent = SimpleNamespace(id="agent-1", instructions="whatever")
    fake_client.create_agent.return_value = new_agent
    captured = {}

    def fake_run_agent(agents_client, agent, content, tool_executor, deadline=None):
        captured["content"] = content
        return "ok"

    monkeypatch.setattr(agents, "_agents_client", lambda: fake_client)
    monkeypatch.setattr(agents, "_find_agent", lambda client, name: None)
    monkeypatch.setattr(agents, "_run_agent", fake_run_agent)
    monkeypatch.setattr(analyzer_module, "curated_frame_urls", lambda a, v, **k: [])

    result = agents.run_analyzer_tools("Describe the scene", "5", video_id="2")

    assert result == "ok"
    assert captured["content"] == (
        "[Context: account_id=5, video_id=2]\n"
        "[No frames have been extracted yet for this video — tell the user to "
        "click \"Extract Frames\" in the console first, rather than guessing "
        "or asking for an upload.]\n"
        "Describe the scene"
    )
    fake_client.create_agent.assert_called_once()


def test_context_includes_curated_frame_urls(monkeypatch):
    import core.azure.analyzer as analyzer_module

    agents = _agents_obj()
    fake_client = MagicMock()
    fake_client.create_agent.return_value = SimpleNamespace(id="agent-1", instructions="x")
    captured = {}

    def fake_run_agent(agents_client, agent, content, tool_executor, deadline=None):
        captured["content"] = content
        return "ok"

    monkeypatch.setattr(agents, "_agents_client", lambda: fake_client)
    monkeypatch.setattr(agents, "_find_agent", lambda client, name: None)
    monkeypatch.setattr(agents, "_run_agent", fake_run_agent)
    monkeypatch.setattr(
        analyzer_module, "curated_frame_urls",
        lambda a, v, **k: ["https://x/frame0.jpg", "https://x/frame1.jpg"],
    )

    agents.run_analyzer_tools("How many land bridges?", "5", video_id="2")

    assert "[Context: account_id=5, video_id=2]" in captured["content"]
    assert "0: https://x/frame0.jpg" in captured["content"]
    assert "1: https://x/frame1.jpg" in captured["content"]
    assert captured["content"].endswith("How many land bridges?")


def test_context_prefix_omitted_without_account_id(monkeypatch):
    agents = _agents_obj()
    fake_client = MagicMock()
    fake_client.create_agent.return_value = SimpleNamespace(id="agent-1", instructions="x")
    captured = {}

    def fake_run_agent(agents_client, agent, content, tool_executor, deadline=None):
        captured["content"] = content
        return "ok"

    monkeypatch.setattr(agents, "_agents_client", lambda: fake_client)
    monkeypatch.setattr(agents, "_find_agent", lambda client, name: None)
    monkeypatch.setattr(agents, "_run_agent", fake_run_agent)

    agents._run_function_agent("raw question", "some-agent", set())

    assert captured["content"] == "raw question"


def test_stale_instructions_are_synced_on_existing_agent(monkeypatch):
    import core.azure.analyzer as analyzer_module

    agents = _agents_obj()
    fake_client = MagicMock()
    existing_agent = SimpleNamespace(id="agent-1", instructions="stale instructions")
    updated_agent = SimpleNamespace(id="agent-1", instructions="new instructions")
    fake_client.update_agent.return_value = updated_agent

    monkeypatch.setattr(agents, "_agents_client", lambda: fake_client)
    monkeypatch.setattr(agents, "_find_agent", lambda client, name: existing_agent)
    monkeypatch.setattr(agents, "_run_agent", lambda *a, **k: "ok")
    monkeypatch.setattr(analyzer_module, "curated_frame_urls", lambda a, v, **k: [])

    agents.run_analyzer_tools("Describe the scene", "5", video_id="2")

    fake_client.update_agent.assert_called_once()
    assert fake_client.update_agent.call_args.args[0] == "agent-1"
    fake_client.create_agent.assert_not_called()


def test_account_video_context_format():
    assert _account_video_context("5", "2") == "[Context: account_id=5, video_id=2]"


def test_run_function_tools_threads_video_id(monkeypatch):
    agents = _agents_obj()
    captured = {}

    def fake_run_function_agent(query_text, agent_name, functions_set,
                                account_id=None, video_id=None, deadline=None):
        captured.update(account_id=account_id, video_id=video_id)
        return "ok"

    monkeypatch.setattr(agents, "_run_function_agent", fake_run_function_agent)

    agents.run_function_tools("q", "5", video_id="2")

    assert captured == {"account_id": "5", "video_id": "2"}


def test_synthesize_from_chat_agent_combines_all_three_team_members(monkeypatch):
    """ask() previously only ever consulted run_analyzer_tools (the CV/tool
    agent), never the knowledge-search agent or the connected function agent
    (Perplexity/Qwen) — so counting/content questions answered from a single
    narrow tool set instead of the whole connected team."""
    agents = _agents_obj()
    monkeypatch.setattr(type(agents), "configured", property(lambda self: False))
    monkeypatch.setattr(agents, "run_connected_agent",
                        lambda q, a, v=None, **k: "SEARCH_MARKER")
    monkeypatch.setattr(agents, "run_function_tools",
                        lambda q, a, v=None, **k: "FUNCTIONS_MARKER")
    monkeypatch.setattr(agents, "run_analyzer_tools",
                        lambda q, a, v=None, **k: "ANALYZER_MARKER")
    monkeypatch.setattr(agents, "_echo", lambda text: text)

    result = agents.synthesize_from_chat_agent("How many land bridges?", "5", video_id="2")

    assert "SEARCH_MARKER" in result
    assert "FUNCTIONS_MARKER" in result
    assert "ANALYZER_MARKER" in result


def test_run_connected_agent_defaults_video_id_and_prefixes_context(monkeypatch, db):
    import azure.search.documents.agent as agent_module
    import azure.search.documents.indexes as indexes_module
    from apps.videos.models import VideoEntity

    video = VideoEntity.objects.create(account_id="5", sas_url="")

    config = SimpleNamespace(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p",
        search_endpoint="https://search.example.net",
        search_admin_key="admin-key",
        search_index_name="dvsa-index",
        search_agent_name="search-agent-in-a-team",
        search_api_version="2025-08-01-preview",
        openai_endpoint="https://oai.example.net",
        openai_api_key="oai-key",
        gpt_deployment="gpt-4o-mini",
        gpt_model="gpt-4o-mini",
    )
    config.search_data_plane_ready = lambda: True
    agents = FoundryAgents(config)

    fake_index_client = MagicMock()
    fake_index_client.list_agents.return_value = []
    fake_index_client.list_knowledge_sources.return_value = []

    fake_retrieval_client = MagicMock()
    fake_response_item = SimpleNamespace(content=[SimpleNamespace(text="search result")])
    fake_retrieval_client.retrieve.return_value = SimpleNamespace(response=[fake_response_item])

    monkeypatch.setattr(indexes_module, "SearchIndexClient", lambda **k: fake_index_client)
    monkeypatch.setattr(agent_module, "KnowledgeAgentRetrievalClient", lambda **k: fake_retrieval_client)

    result = agents.run_connected_agent("How many land bridges?", "5")

    assert result == "search result"
    sent_request = fake_retrieval_client.retrieve.call_args.kwargs["retrieval_request"]
    sent_text = sent_request.messages[0].content[0].text
    assert sent_text == (
        f"[Context: account_id=5, video_id={video.id}]\nHow many land bridges?"
    )
    # Scoped to the account AND the resolved video (not just account-wide).
    filter_add_on = sent_request.target_index_params[0].filter_add_on
    assert filter_add_on == (
        f"account_id eq '5' and startswith(path, '5/images/{video.id}/')"
    )
    # No api_version override — the pinned SDK rejects c.search_api_version.
    assert "api_version" not in fake_retrieval_client.retrieve.call_args.kwargs


def test_run_connected_agent_strips_openai_v1_suffix(monkeypatch, db):
    import azure.search.documents.agent as agent_module
    import azure.search.documents.indexes as indexes_module

    config = SimpleNamespace(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p",
        search_endpoint="https://search.example.net",
        search_admin_key="admin-key",
        search_index_name="dvsa-index",
        search_agent_name="search-agent-in-a-team",
        search_api_version="2025-08-01-preview",
        # The unified "/openai/v1" endpoint shape — must be stripped before
        # being handed to Search as the AOAI resourceUri.
        openai_endpoint="https://found-vision-1.openai.azure.com/openai/v1",
        openai_api_key="oai-key",
        gpt_deployment="gpt-4o-mini",
        gpt_model="gpt-4o-mini",
    )
    config.search_data_plane_ready = lambda: True
    agents = FoundryAgents(config)

    fake_index_client = MagicMock()
    fake_retrieval_client = MagicMock()
    fake_retrieval_client.retrieve.return_value = SimpleNamespace(
        response=[SimpleNamespace(content=[SimpleNamespace(text="ok")])]
    )
    monkeypatch.setattr(indexes_module, "SearchIndexClient", lambda **k: fake_index_client)
    monkeypatch.setattr(agent_module, "KnowledgeAgentRetrievalClient", lambda **k: fake_retrieval_client)

    agents.run_connected_agent("q", "5", video_id="2")

    sent_agent = fake_index_client.create_or_update_agent.call_args.kwargs["agent"]
    resource_url = sent_agent.models[0].azure_open_ai_parameters.resource_url
    assert resource_url == "https://found-vision-1.openai.azure.com"


def test_run_connected_agent_failure_degrades_to_none(monkeypatch):
    """A Search/model-side failure must not crash the whole chat synthesis —
    run_function_tools/run_analyzer_tools can still answer."""
    config = SimpleNamespace(
        project_endpoint="https://example.services.ai.azure.com/api/projects/p",
        search_endpoint="https://search.example.net",
        search_admin_key="admin-key",
        search_index_name="dvsa-index",
    )
    config.search_data_plane_ready = lambda: True
    agents = FoundryAgents(config)

    def boom(**kwargs):
        raise RuntimeError("Search is unreachable")

    import azure.search.documents.indexes as indexes_module

    monkeypatch.setattr(indexes_module, "SearchIndexClient", boom)

    assert agents.run_connected_agent("q", "5") is None


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
