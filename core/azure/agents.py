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

import logging
import time
from typing import Any, Callable, List, Optional

from .config import AzureEnvironmentConfig

logger = logging.getLogger("apps.azure")

_AGENT_MAX_OUTPUT_TOKENS = 10000


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
                        output = tool_executor(tool_call)
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
    def _run_function_agent(self, query_text, agent_name, functions_set) -> Optional[str]:
        from azure.ai.agents.models import (  # noqa: PLC0415
            FunctionTool, RequiredFunctionToolCall,
        )

        agents_client = self._agents_client()
        functions = FunctionTool(functions=functions_set)
        instructions = (
            "You are a drone aerial image analytics assistant that answers the "
            "question by finding a suitable function, passing the question to it, "
            "evaluating it and relaying the response. If you can't find a suitable "
            "function, default to the ask_perplexity function in your tools."
        )
        with agents_client:
            agent = self._find_agent(agents_client, agent_name)
            if agent is None:
                agent = agents_client.create_agent(
                    model=self.config.agent_model, name=agent_name,
                    instructions=instructions, tools=functions.definitions,
                    tool_resources=functions.resources, top_p=1,
                )

            def _exec(tool_call):
                if isinstance(tool_call, RequiredFunctionToolCall):
                    return functions.execute(tool_call)
                return None

            return self._run_agent(agents_client, agent, query_text, _exec)

    def run_function_tools(self, query_text, account_id) -> Optional[str]:
        if not self.configured:
            return self._echo(query_text)
        from .analyzer import image_user_functions  # noqa: PLC0415

        return self._run_function_agent(query_text, self.config.fn_agent_name, image_user_functions())

    def run_analyzer_tools(self, query_text, account_id) -> Optional[str]:
        if not self.configured:
            return self._echo(query_text)
        from .analyzer import analyzer_functions  # noqa: PLC0415

        return self._run_function_agent(query_text, self.config.tool_agent_name, analyzer_functions())

    # ----- AI-search knowledge agent ------------------------------------
    def run_connected_agent(self, query_text, account_id, index_name=None) -> Optional[str]:
        """Create/reuse a KnowledgeAgent over the index and retrieve an answer."""
        if not self.configured or not self.config.search_data_plane_ready():
            return self._echo(query_text)
        from azure.core.credentials import AzureKeyCredential  # noqa: PLC0415
        from azure.search.documents.indexes import SearchIndexClient  # noqa: PLC0415
        from azure.search.documents.indexes.models import (  # noqa: PLC0415
            AzureOpenAIVectorizerParameters, KnowledgeAgent,
            KnowledgeAgentAzureOpenAIModel, KnowledgeAgentOutputConfiguration,
            KnowledgeAgentOutputConfigurationModality, KnowledgeAgentRequestLimits,
            KnowledgeSourceReference, SearchIndexKnowledgeSource,
            SearchIndexKnowledgeSourceParameters,
        )
        from azure.search.documents.agent import KnowledgeAgentRetrievalClient  # noqa: PLC0415
        from azure.search.documents.agent.models import (  # noqa: PLC0415
            KnowledgeAgentMessage, KnowledgeAgentMessageTextContent,
            KnowledgeAgentRetrievalRequest,
        )

        c = self.config
        index_name = index_name or c.search_index_name
        cred = AzureKeyCredential(c.search_admin_key)
        index_client = SearchIndexClient(endpoint=c.search_endpoint, credential=cred)
        agent = next((a for a in index_client.list_agents() if a.name == c.search_agent_name), None)
        if agent is None:
            if not any(s.name == index_name for s in index_client.list_knowledge_sources()):
                index_client.create_knowledge_source(
                    knowledge_source=SearchIndexKnowledgeSource(
                        name=index_name,
                        search_index_parameters=SearchIndexKnowledgeSourceParameters(
                            search_index_name=index_name,
                            source_data_select="id,account_id,description,location,created",
                        ),
                    ),
                    api_version=c.search_api_version,
                )
            model = KnowledgeAgentAzureOpenAIModel(
                azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                    resource_url=c.openai_endpoint, deployment_name=c.gpt_deployment,
                    model_name=c.gpt_model, api_key=c.openai_api_key,
                )
            )
            agent = KnowledgeAgent(
                name=c.search_agent_name, models=[model],
                knowledge_sources=[KnowledgeSourceReference(
                    name=index_name, include_references=True,
                    include_reference_source_data=False, always_query_soure=True,
                    max_sub_queries=10, reranker_threshold=2.5,
                )],
                request_limits=KnowledgeAgentRequestLimits(max_output_size=_AGENT_MAX_OUTPUT_TOKENS),
                retrieval_instructions=(
                    "You are an aerial drone image analyst. If an account_id is "
                    "provided with the query, select only images whose account_id "
                    "matches and answer from those images and their vectors/fields."
                ),
                output_configuration=KnowledgeAgentOutputConfiguration(
                    modality=KnowledgeAgentOutputConfigurationModality.ANSWER_SYNTHESIS,
                    include_activity=True,
                ),
            )
            index_client.create_or_update_agent(agent=agent)

        retrieval_client = KnowledgeAgentRetrievalClient(
            endpoint=c.search_endpoint, agent_name=c.search_agent_name, credential=cred
        )
        req = KnowledgeAgentRetrievalRequest(messages=[
            KnowledgeAgentMessage(role="user", content=[KnowledgeAgentMessageTextContent(text=query_text)])
        ])
        result = retrieval_client.retrieve(retrieval_request=req, api_version=c.search_api_version)
        return result.response[0].content[0].text

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

    def synthesize_from_chat_agent(self, query_text, account_id) -> str:
        """Consolidate analyzer-tool output into a smooth narrative via chat agent."""
        delegated = self.run_analyzer_tools(query_text, account_id)
        synthesis = f"[User]: {query_text}\n\n[Connected Agent Output]:\n{delegated}\n"
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
