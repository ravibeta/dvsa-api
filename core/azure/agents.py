"""Azure AI Foundry agentic runtime (ported from ezvision videos/myvideoanalyzer.py).

A :class:`FoundryAgents` wraps the Foundry/Agents SDKs (``azure.ai.projects``,
``azure.ai.agents``, ``azure.search.documents`` knowledge agents) to:

- run an Azure AI Search *knowledge agent* over the aerial-frame index
  (:meth:`run_connected_agent`, :meth:`knowledge_base_search`),
- run *function-tool* agents that call the ported analyzer functions
  (:meth:`run_function_tools`, :meth:`run_analyzer_tools`),
- synthesize a final narrative (:meth:`synthesize_from_chat_agent`,
  :meth:`synthesize_from_agents`), and
- resolve object/scene URIs (:meth:`object_in_scene_search`).

The shared agent run-loop (create thread → message → run → submit tool outputs →
collect answer) is factored into :meth:`_run_agent`. When the Foundry project is
not configured, every public method returns a deterministic echo answer via
``apps.observability.llm`` so the pipeline stays runnable offline.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, List, Optional

from .config import AzureEnvironmentConfig

logger = logging.getLogger("apps.azure")

_AGENT_MAX_OUTPUT_TOKENS = 10000


def _json_safe(o: Any) -> Any:
    """JSON ``default`` hook: coerce non-native objects (notably NumPy arrays
    and scalars) into JSON-serializable values."""
    tolist = getattr(o, "tolist", None)  # numpy.ndarray
    if callable(tolist):
        return tolist()
    item = getattr(o, "item", None)  # numpy scalar
    if callable(item):
        return item()
    return str(o)


def _coerce_tool_output(output: Any) -> Optional[str]:
    """Foundry ``submit_tool_outputs`` JSON-encodes every ToolOutput, which
    throws on a NumPy ``ndarray`` returned by an analyzer function. Normalize
    any tool return value into a JSON-safe string (strings/None pass through)."""
    if output is None or isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=_json_safe)
    except TypeError:
        return str(output)


def _account_video_context(account_id: Optional[str], video_id: Optional[str]) -> str:
    """The ``[Context: ...]`` line every team member's prompt is prefixed
    with, so it knows which account/video to scope its tools/search to."""
    return f"[Context: account_id={account_id}, video_id={video_id}]"


class FoundryAgents:
    def __init__(self, config: AzureEnvironmentConfig) -> None:
        self.config = config

    @property
    def configured(self) -> bool:
        return bool(self.config.project_endpoint)

    def _echo(self, query_text: str) -> str:
        """Offline fallback answer."""
        try:
            from apps.observability.llm import get_llm_client  # noqa: PLC0415

            return get_llm_client().complete(
                query_text, system="You are an aerial drone image analyst."
            )
        except Exception:  # noqa: BLE001
            return f"[offline] {query_text}"

    # ----- SDK clients (lazy) -------------------------------------------
    def _credential(self):
        from azure.identity import DefaultAzureCredential  # noqa: PLC0415

        return DefaultAzureCredential()

    def _project_client(self):
        from azure.ai.projects import AIProjectClient  # noqa: PLC0415

        return AIProjectClient(endpoint=self.config.project_endpoint, credential=self._credential())

    def _agents_client(self):
        from azure.ai.agents import AgentsClient  # noqa: PLC0415

        return AgentsClient(endpoint=self.config.project_endpoint, credential=self._credential())

    # ----- low-level agent lookups --------------------------------------
    def get_agent_id(self, agent_name: str) -> Optional[str]:
        client = self._agents_client()
        for entry in client.list_agents():
            if entry.name == agent_name:
                return entry.id
        logger.warning("Agent not found: %s", agent_name)
        return None

    def _find_agent(self, agents_client, name):
        for entity in agents_client.list_agents():
            if entity.name == name:
                return entity
        return None

    def ask_agent(self, agent_name: str, query_text: str):
        """Run a simple thread against a named agent; return its messages."""
        if not self.configured:
            return None
        agent_id = self.get_agent_id(agent_name)
        if not agent_id:
            return None
        from azure.ai.agents.models import ListSortOrder  # noqa: PLC0415

        pc = self._project_client()
        thread = pc.agents.threads.create()
        pc.agents.messages.create(thread_id=thread.id, role="user", content=query_text)
        run = pc.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent_id)
        if run.status == "failed":
            logger.error("Run failed: %s", run.last_error)
            return None
        return pc.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)

    def ask_agent_for_url(self, agent_name: str, query_text: str) -> Optional[str]:
        messages = self.ask_agent(agent_name, query_text)
        if not messages:
            return None
        for message in messages:
            if message.text_messages:
                for item in message.text_messages:
                    if item and item.text:
                        for annotation in item.text.annotations:
                            if annotation.type == "url_citation":
                                return annotation.url_citation.url
        return None

    def delete_all_threads_for_agent(self, agent_name: str) -> None:
        if not self.configured:
            return
        agents_client = self._agents_client()
        for thread in agents_client.threads.list():
            agents_client.threads.delete(thread_id=thread.id)
            logger.info("Deleted thread ID: %s", thread.id)

    # ----- shared run-loop ----------------------------------------------
    def _run_agent(self, agents_client, agent, content: str,
                   tool_executor: Optional[Callable[[Any], Optional[str]]] = None) -> Optional[str]:
        """Create a thread, run ``agent`` over ``content``, return its answer.

        ``tool_executor(tool_call) -> output|None`` handles a single tool call
        (function / AI-search / OpenAPI). Mirrors the source run loops.
        """
        from azure.ai.agents.models import (  # noqa: PLC0415
            ListSortOrder, SubmitToolOutputsAction, ToolOutput,
        )

        answer = None
        thread = agents_client.threads.create()
        agents_client.messages.create(thread_id=thread.id, role="user", content=content)
        run = agents_client.runs.create(thread_id=thread.id, agent_id=agent.id)
        while run.status in ("queued", "in_progress", "requires_action"):
            time.sleep(1)
            run = agents_client.runs.get(thread_id=thread.id, run_id=run.id)
            if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                if not tool_calls:
                    agents_client.runs.cancel(thread_id=thread.id, run_id=run.id)
                    break
                tool_outputs = []
                for tool_call in tool_calls:
                    if tool_executor is None:
                        continue
                    try:
                        output = _coerce_tool_output(tool_executor(tool_call))
                    except Exception as exc:  # noqa: BLE001
                        logger.info("Error executing tool_call %s: %s", tool_call.id, exc)
                        continue
                    if output is not None:
                        answer = output
                        tool_outputs.append(ToolOutput(tool_call_id=tool_call.id, output=output))
                if tool_outputs:
                    agents_client.runs.submit_tool_outputs(
                        thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                    )
        messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for msg in messages:
            if msg.text_messages:
                answer = msg.text_messages[-1].text.value
        return answer

    # ----- function-tool agents -----------------------------------------
    def _run_function_agent(self, query_text, agent_name, functions_set,
                            account_id=None, video_id=None) -> Optional[str]:
        from azure.ai.agents.models import (  # noqa: PLC0415
            FunctionTool, RequiredFunctionToolCall,
        )

        agents_client = self._agents_client()
        functions = FunctionTool(functions=functions_set)
        instructions = (
            "You are a drone aerial image analytics assistant that answers the "
            "question by finding a suitable function, passing the question to it, "
            "evaluating it and relaying the response. For questions about what is "
            "visible in the scene — description, notable objects/events, or "
            "counting things by appearance (e.g. how many railway tracks or land "
            "bridges) — call describe_frame for *every* extracted frame URL/index "
            "listed in the context below, not just the first one, and combine what "
            "you see across all of them before answering; a feature may only be "
            "visible in some frames. Only if describe_frame can't answer, fall back "
            "to ask_perplexity with its frames_list argument set to the same "
            "extracted frame indices from the context. "
            "Every user message is preceded by a '[Context: account_id=..., "
            "video_id=...]' line and, when frames have been extracted for that "
            "video, a list of their real URLs/indices — always pass those exact "
            "account_id/video_id values and those exact frame URLs/indices to your "
            "tools (get_sas_url_template, describe_frame, ask_perplexity, "
            "agentic_retrieval, ...). Never ask the user to upload an image or "
            "supply a URL, and never invent or guess an image URL or frame number "
            "yourself."
        )
        with agents_client:
            agent = self._find_agent(agents_client, agent_name)
            if agent is None:
                agent = agents_client.create_agent(
                    model=self.config.agent_model, name=agent_name,
                    instructions=instructions, tools=functions.definitions,
                    tool_resources=functions.resources, top_p=1,
                )
            elif agent.instructions != instructions:
                # Keep a previously-created agent's instructions in sync (e.g.
                # agents created before this grounding context was added).
                agent = agents_client.update_agent(agent.id, instructions=instructions)

            def _exec(tool_call):
                if isinstance(tool_call, RequiredFunctionToolCall):
                    return functions.execute(tool_call)
                return None

            content = query_text
            if account_id:
                from .analyzer import curated_frame_urls  # noqa: PLC0415

                lines = [_account_video_context(account_id, video_id)]
                frame_urls = curated_frame_urls(account_id, video_id)
                if frame_urls:
                    lines.append(
                        "[Extracted frames for this video — pass one of these URLs "
                        "directly to download_image/describe_frame, or their "
                        "0-based indices as a comma-separated frames_list to "
                        "ask_perplexity (e.g. \"0,1,2\"). Check all of them, not "
                        "just the first.]"
                    )
                    lines.extend(f"{i}: {url}" for i, url in enumerate(frame_urls))
                else:
                    lines.append(
                        "[No frames have been extracted yet for this video — tell "
                        "the user to click \"Extract Frames\" in the console "
                        "first, rather than guessing or asking for an upload.]"
                    )
                content = "\n".join(lines) + f"\n{query_text}"
            return self._run_agent(agents_client, agent, content, _exec)

    def run_function_tools(self, query_text, account_id, video_id=None) -> Optional[str]:
        if not self.configured:
            return self._echo(query_text)
        from .analyzer import image_user_functions  # noqa: PLC0415

        return self._run_function_agent(query_text, self.config.fn_agent_name, image_user_functions(),
                                        account_id=account_id, video_id=video_id)

    def run_analyzer_tools(self, query_text, account_id, video_id=None) -> Optional[str]:
        if not self.configured:
            return self._echo(query_text)
        from .analyzer import analyzer_functions  # noqa: PLC0415

        return self._run_function_agent(query_text, self.config.tool_agent_name, analyzer_functions(),
                                        account_id=account_id, video_id=video_id)

    # ----- AI-search knowledge agent ------------------------------------
    def run_connected_agent(self, query_text, account_id, video_id=None,
                            index_name=None) -> Optional[str]:
        """Create/reuse a KnowledgeAgent over the index and retrieve an answer.

        Scoped to one video: when ``video_id`` is not given it defaults to the
        account's latest :class:`~apps.videos.models.VideoEntity`, matching
        :func:`core.azure.analyzer.get_sas_url_template`'s convention. Scoping
        uses a structured OData ``filter_add_on`` (account_id, plus the frame
        blob path's ``video_id`` segment) — enforced by the index itself,
        rather than asking the retrieval model to interpret free-text
        instructions.

        Targets indexes directly (``KnowledgeAgentTargetIndex``/
        ``KnowledgeAgentIndexParams``): the pinned ``azure-search-documents``
        (11.6.0b12) predates the separate "knowledge source" resource some
        newer preview SDKs expose, so that abstraction isn't available here.
        """
        if not self.configured or not self.config.search_data_plane_ready():
            return self._echo(query_text)
        if not video_id:
            try:
                from apps.videos.models import VideoEntity  # noqa: PLC0415

                video_id = str(VideoEntity.objects.filter(account_id=account_id).last().id)
            except Exception as exc:  # noqa: BLE001
                logger.info("run_connected_agent: could not resolve latest video: %s", exc)
                video_id = None
        from azure.core.credentials import AzureKeyCredential  # noqa: PLC0415
        from azure.search.documents.indexes import SearchIndexClient  # noqa: PLC0415
        from azure.search.documents.indexes.models import (  # noqa: PLC0415
            AzureOpenAIVectorizerParameters, KnowledgeAgent,
            KnowledgeAgentAzureOpenAIModel, KnowledgeAgentRequestLimits,
            KnowledgeAgentTargetIndex,
        )
        from azure.search.documents.agent import KnowledgeAgentRetrievalClient  # noqa: PLC0415
        from azure.search.documents.agent.models import (  # noqa: PLC0415
            KnowledgeAgentIndexParams, KnowledgeAgentMessage,
            KnowledgeAgentMessageTextContent, KnowledgeAgentRetrievalRequest,
        )

        c = self.config
        index_name = index_name or c.search_index_name
        try:
            cred = AzureKeyCredential(c.search_admin_key)
            index_client = SearchIndexClient(endpoint=c.search_endpoint, credential=cred)
            # Azure AI Search builds its own
            # "{resourceUri}/openai/deployments/{deploymentId}/..." path (the
            # same convention _azure_openai_embedding uses), so resourceUri
            # must be the bare resource host. c.openai_endpoint may carry the
            # newer unified "/openai/v1" suffix (e.g. for the openai SDK) —
            # strip it defensively rather than 404 against the model.
            resource_url = (c.openai_endpoint or "").rstrip("/")
            if resource_url.lower().endswith("/openai/v1"):
                resource_url = resource_url[: -len("/openai/v1")]
            model = KnowledgeAgentAzureOpenAIModel(
                azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                    resource_url=resource_url, deployment_name=c.gpt_deployment,
                    model_name=c.gpt_model, api_key=c.openai_api_key,
                )
            )
            agent = KnowledgeAgent(
                name=c.search_agent_name, models=[model],
                target_indexes=[KnowledgeAgentTargetIndex(
                    index_name=index_name, default_include_reference_source_data=True,
                    default_reranker_threshold=2.5,
                )],
                request_limits=KnowledgeAgentRequestLimits(max_output_size=_AGENT_MAX_OUTPUT_TOKENS),
            )
            # create_or_update_agent is an idempotent upsert — call it every
            # time so the agent's model/target-index config always reflects
            # the current code/env rather than staying frozen at whatever it
            # was the first time this ran.
            index_client.create_or_update_agent(agent=agent)

            retrieval_client = KnowledgeAgentRetrievalClient(
                endpoint=c.search_endpoint, agent_name=c.search_agent_name, credential=cred
            )
            content_text = query_text
            target_index_params = None
            if account_id:
                content_text = f"{_account_video_context(account_id, video_id)}\n{query_text}"
                filter_add_on = f"account_id eq '{account_id}'"
                if video_id:
                    filter_add_on += f" and startswith(path, '{account_id}/images/{video_id}/')"
                target_index_params = [KnowledgeAgentIndexParams(
                    index_name=index_name, filter_add_on=filter_add_on,
                )]
            req = KnowledgeAgentRetrievalRequest(
                messages=[KnowledgeAgentMessage(
                    role="user", content=[KnowledgeAgentMessageTextContent(text=content_text)],
                )],
                target_index_params=target_index_params,
            )
            # No explicit api_version override here: the pinned
            # azure-search-documents (11.6.0b12) retrieval REST layer only
            # recognizes its own default (DEFAULT_VERSION, currently
            # 2025-05-01-preview) — c.search_api_version (2025-08-01-preview)
            # is for a newer preview and gets rejected with "The version
            # indicated by the api-version query string parameter does not
            # exist."
            result = retrieval_client.retrieve(retrieval_request=req)
            return result.response[0].content[0].text
        except Exception as exc:  # noqa: BLE001
            # A knowledge-agent-side failure (Search/model misconfiguration,
            # transient outage, ...) must not take down the whole synthesis —
            # run_function_tools/run_analyzer_tools can still answer.
            logger.warning("run_connected_agent failed: %s", exc)
            return None

    def knowledge_base_search(self, query_text, account_id) -> Optional[str]:
        """Search-tool agent with vector-semantic-hybrid filter on account_id."""
        if not self.configured:
            return self._echo(query_text)
        from azure.ai.agents.models import (  # noqa: PLC0415
            AzureAISearchQueryType, AzureAISearchTool, ConnectedAgentTool,
            RunStepAzureAISearchToolCall,
        )

        c = self.config
        agents_client = self._agents_client()
        project_client = self._project_client()
        connected_agent = self._find_agent(project_client.agents, c.fn_agent_name) \
            if hasattr(project_client, "agents") else None
        ai_search_tool = AzureAISearchTool(
            index_connection_id=c.search_connection_id, index_name=c.search_index_name,
            query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID, top_k=3,
            filter=f"account_id eq '{account_id}'",
        )
        instructions = (
            "You are a drone aerial image analytics assistant that answers by "
            "searching an Azure AI Search index or delegating to a connected agent, "
            "then synthesizing a comprehensive response. If none, reply 'I do not know.'"
        )
        with agents_client:
            agent = self._find_agent(agents_client, "master-agent-in-a-team")
            tools = ai_search_tool.definitions
            resources = ai_search_tool.resources
            if connected_agent is not None:
                connected_tool = ConnectedAgentTool(
                    id=connected_agent.id, name="connected_agent",
                    description="Delegate to the function agent when search is inconclusive.",
                )
                tools = tools + connected_tool.definitions
                resources = resources + connected_tool.resources
            if agent is None:
                agent = agents_client.create_agent(
                    model="gpt-4o-mini", name="master-agent-in-a-team",
                    instructions=instructions, tools=tools, tool_resources=resources, top_p=1,
                )

            def _exec(tool_call):
                if isinstance(tool_call, RunStepAzureAISearchToolCall):
                    return ai_search_tool.execute(tool_call)
                return None

            return self._run_agent(agents_client, agent, query_text, _exec)

    # ----- synthesis -----------------------------------------------------
    def synthesize_from_agents(self, query_text, account_id) -> str:
        knowledge = self.knowledge_base_search(query_text, account_id)
        delegated = self.run_function_tools(query_text, account_id)
        return (
            f"\n[Search Agent Output]:\n{knowledge}\n\n"
            f"[Connected Agent Output]:\n{delegated}\n"
        )

    def synthesize_from_chat_agent(self, query_text, account_id, video_id=None) -> str:
        """Consolidate the connected team's output into a smooth narrative.

        Runs all three real team members — the native AI Search knowledge
        agent (vector search + automatic query decomposition over every
        extracted frame's stored caption/tags/vector), the connected function
        agent (Perplexity/Qwen/agentic_retrieval), and the CV/analyzer tool
        agent — then has ``chat-agent-in-a-team`` narrate across all three.
        """
        search = self.run_connected_agent(query_text, account_id, video_id)
        functions = self.run_function_tools(query_text, account_id, video_id)
        analyzer = self.run_analyzer_tools(query_text, account_id, video_id)
        synthesis = (
            f"[User]: {query_text}\n\n"
            f"[Knowledge Search Agent Output]:\n{search}\n\n"
            f"[Connected Function Agent Output]:\n{functions}\n\n"
            f"[Tool Agent Output]:\n{analyzer}\n"
        )
        if not self.configured:
            return self._echo(synthesis)
        from azure.ai.agents.models import (  # noqa: PLC0415
            OpenApiConnectionAuthDetails, OpenApiTool, RunStepOpenAPIToolCall,
        )

        agents_client = self._agents_client()
        instructions = (
            "You are an aerial drone image analyst who consolidates and rephrases "
            "answers from other agents into a smooth narrative without bullet points, "
            "fulfilling the user's question in one attempt without clarifying questions."
        )
        with agents_client:
            agent = self._find_agent(agents_client, self.config.chat_agent_name)
            if agent is None:
                api = OpenApiTool(name=self.config.chat_agent_name,
                                  description="consolidator of answers", spec={},
                                  auth=OpenApiConnectionAuthDetails())
                agent = agents_client.create_agent(
                    model=self.config.agent_model, name=self.config.chat_agent_name,
                    instructions=instructions, tools=api.definitions,
                    tool_resources=api.resources, top_p=1,
                )

                def _exec(tool_call):
                    if isinstance(tool_call, RunStepOpenAPIToolCall):
                        return api.execute(tool_call)
                    return None

                answer = self._run_agent(agents_client, agent, synthesis, _exec)
            else:
                answer = self._run_agent(agents_client, agent, synthesis, None)
        return answer or synthesis

    def file_agent_search(self, query_text, account_id) -> Optional[str]:
        """Alias of the master search agent (source file_agent_search)."""
        return self.knowledge_base_search(query_text, account_id)

    def object_in_scene_search(self, query_text, account_id, video_id=None) -> Optional[str]:
        from .analyzer import ask_perplexity, get_object_uri, get_scene_uri  # noqa: PLC0415

        object_uri = get_object_uri(query_text, account_id, video_id)
        scene_uri = get_scene_uri(query_text, account_id, video_id)
        if object_uri and scene_uri:
            q = (f"How many objects given by image URI {object_uri} are found in the "
                 f"image given by image URI {scene_uri} where objects are described in: {query_text}?")
            return ask_perplexity(q, account_id=account_id, video_id=video_id)
        return None
