"""Configuration contract for the session-scoped Azure environment.

A single immutable :class:`AzureEnvironmentConfig` collects every value the
provisioning and data-plane code needs. It can be built from Django settings
(:meth:`AzureEnvironmentConfig.from_settings`) or directly from process
environment variables (:meth:`AzureEnvironmentConfig.from_env`) so the package
is usable from management commands, Celery tasks, and plain scripts alike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name, default)
    return val if val not in ("", None) else default


@dataclass(frozen=True)
class AzureEnvironmentConfig:
    """Resolved configuration for one Azure environment.

    Defaults mirror ``config/settings/base.py`` so the two never drift; the
    storage account name (``sadronevideo``) and the 1536-dim vector width are
    the contract requirements from the spec.
    """

    # Control plane (ARM)
    subscription_id: Optional[str] = None
    resource_group: str = "rg-dvsa"
    location: str = "eastus"
    provisioner: str = "auto"  # auto | sdk | terraform | dryrun
    terraform_dir: Optional[str] = None

    # Storage
    storage_account: str = "sadronevideo"
    storage_connection_string: Optional[str] = None
    account_key: Optional[str] = None
    input_container: str = "input"
    output_container: str = "output"

    # AI Search
    search_service_name: Optional[str] = None
    search_endpoint: Optional[str] = None
    search_admin_key: Optional[str] = None
    search_index_name: str = "dvsa-index"
    search_sku: str = "standard"
    vector_dimensions: int = 1536

    # Azure OpenAI / AI Foundry
    openai_endpoint: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_account: str = "dvsa-foundry"
    gpt_deployment: str = "gpt-4o-mini"
    gpt_model: str = "gpt-4o-mini"
    embedding_deployment: str = "text-embedding-ada-002"
    embedding_model: str = "text-embedding-ada-002"
    openai_api_version: str = "2024-06-01"

    # Session isolation
    session_isolation: str = "index"  # index | filter

    # --- Runtime data-plane (ported from ezvision) -----------------------
    vision_api_version: str = "2024-02-01"
    search_api_version: str = "2025-08-01-preview"
    vision_model_version: str = "2023-04-15"

    # Foundry project (agents live here) + agent role names.
    project_endpoint: Optional[str] = None
    project_api_key: Optional[str] = None
    agent_model: str = "gpt-4o-mini"
    fn_agent_name: str = "fn-agent-in-a-team"
    chat_agent_name: str = "chat-agent-in-a-team"
    search_agent_name: str = "search-agent-in-a-team"
    tool_agent_name: str = "tool-agent-in-a-team"
    # AI Search coordinates for agentic retrieval.
    search_connection_id: Optional[str] = None
    search_resource_id: Optional[str] = None
    search_subscription: Optional[str] = None
    search_resource_group: Optional[str] = None
    search_location: str = "eastus"

    # Video Indexer.
    video_indexer_url: str = "https://api.videoindexer.ai"
    video_indexer_region: str = "eastus"
    video_indexer_account: Optional[str] = None
    video_indexer_api_key: Optional[str] = None
    video_indexer_access_token: str = ""

    # Perplexity.
    perplexity_chat_api_key: Optional[str] = None
    perplexity_chat_api_url: str = "https://api.perplexity.ai/chat/completions"
    perplexity_geo_api_key: Optional[str] = None
    perplexity_geo_api_url: str = "https://api.perplexity.ai/v1/image/geolocation"

    # Sample object/scene URIs.
    sample_object_uri: str = ""
    sample_scene_uri: str = ""

    @property
    def agent_names(self) -> tuple:
        """Foundry agent roles (back-compat tuple used by provisioning)."""
        return (
            self.fn_agent_name,
            self.chat_agent_name,
            self.search_agent_name,
            self.tool_agent_name,
        )

    # ----- builders ------------------------------------------------------
    @classmethod
    def from_settings(cls, settings=None) -> "AzureEnvironmentConfig":
        """Build from Django settings, falling back to env when no Django."""
        if settings is None:
            try:
                from django.conf import settings as dj_settings

                settings = dj_settings
            except Exception:  # noqa: BLE001 - no Django context
                return cls.from_env()

        def g(name, default=None):
            return getattr(settings, name, default)

        return cls(
            subscription_id=g("AZURE_SUBSCRIPTION_ID"),
            resource_group=g("AZURE_RESOURCE_GROUP", "rg-dvsa"),
            location=g("AZURE_LOCATION", "eastus"),
            provisioner=g("AZURE_PROVISIONER", "auto"),
            terraform_dir=g("AZURE_TERRAFORM_DIR"),
            storage_account=g("AZURE_STORAGE_ACCOUNT", "sadronevideo"),
            storage_connection_string=g("AZURE_STORAGE_CONNECTION_STRING"),
            account_key=g("AZURE_ACCOUNT_KEY"),
            input_container=g("AZURE_STORAGE_INPUT_CONTAINER", "input"),
            output_container=g("AZURE_STORAGE_OUTPUT_CONTAINER", "output"),
            search_service_name=g("AZURE_SEARCH_SERVICE_NAME"),
            search_endpoint=g("AZURE_SEARCH_ENDPOINT"),
            search_admin_key=g("AZURE_SEARCH_ADMIN_KEY"),
            search_index_name=g("AZURE_SEARCH_INDEX_NAME", "dvsa-index"),
            search_sku=g("AZURE_SEARCH_SKU", "standard"),
            vector_dimensions=int(g("AZURE_SEARCH_VECTOR_DIMENSIONS", 1536)),
            openai_endpoint=g("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=g("AZURE_OPENAI_API_KEY"),
            openai_account=g("AZURE_OPENAI_ACCOUNT", "dvsa-foundry"),
            gpt_deployment=g("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-mini"),
            gpt_model=g("AZURE_OPENAI_GPT_MODEL", "gpt-4o-mini"),
            embedding_deployment=g(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
            ),
            embedding_model=g(
                "AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"
            ),
            openai_api_version=g("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            session_isolation=g("AZURE_SESSION_ISOLATION", "index"),
            vision_api_version=g("AZURE_VISION_API_VERSION", "2024-02-01"),
            search_api_version=g("AZURE_SEARCH_API_VERSION", "2025-08-01-preview"),
            vision_model_version=g("AZURE_VISION_MODEL_VERSION", "2023-04-15"),
            project_endpoint=g("AZURE_PROJECT_ENDPOINT"),
            project_api_key=g("AZURE_PROJECT_API_KEY"),
            agent_model=g("AZURE_AGENT_MODEL", "gpt-4o-mini"),
            fn_agent_name=g("AZURE_FN_AGENT_NAME", "fn-agent-in-a-team"),
            chat_agent_name=g("AZURE_CHAT_AGENT_NAME", "chat-agent-in-a-team"),
            search_agent_name=g("AZURE_SEARCH_AGENT_NAME", "search-agent-in-a-team"),
            tool_agent_name=g("AZURE_TOOL_AGENT_NAME", "tool-agent-in-a-team"),
            search_connection_id=g("AZURE_SEARCH_CONNECTION_ID"),
            search_resource_id=g("AZURE_SEARCH_RESOURCE_ID"),
            search_subscription=g("AZURE_SEARCH_SUBSCRIPTION"),
            search_resource_group=g("AZURE_SEARCH_RESOURCE_GROUP"),
            search_location=g("AZURE_SEARCH_LOCATION", "eastus"),
            video_indexer_url=g("AZURE_VIDEO_INDEXER_URL", "https://api.videoindexer.ai"),
            video_indexer_region=g("AZURE_VIDEO_INDEXER_REGION", "eastus"),
            video_indexer_account=g("AZURE_VIDEO_INDEXER_ACCOUNT"),
            video_indexer_api_key=g("AZURE_VIDEO_INDEXER_API_KEY"),
            video_indexer_access_token=g("AZURE_VIDEO_INDEXER_ACCESS_TOKEN", ""),
            perplexity_chat_api_key=g("PERPLEXITY_CHAT_API_KEY"),
            perplexity_chat_api_url=g(
                "PERPLEXITY_CHAT_API_URL", "https://api.perplexity.ai/chat/completions"
            ),
            perplexity_geo_api_key=g("PERPLEXITY_GEO_API_KEY"),
            perplexity_geo_api_url=g(
                "PERPLEXITY_GEO_API_URL",
                "https://api.perplexity.ai/v1/image/geolocation",
            ),
            sample_object_uri=g("SAMPLE_OBJECT_URI", ""),
            sample_scene_uri=g("SAMPLE_SCENE_URI", ""),
        )

    @classmethod
    def from_env(cls) -> "AzureEnvironmentConfig":
        """Build directly from ``os.environ`` (no Django required)."""
        return cls(
            subscription_id=_env("AZURE_SUBSCRIPTION_ID"),
            resource_group=_env("AZURE_RESOURCE_GROUP", "rg-dvsa"),
            location=_env("AZURE_LOCATION", "eastus"),
            provisioner=_env("AZURE_PROVISIONER", "auto"),
            terraform_dir=_env("AZURE_TERRAFORM_DIR"),
            storage_account=_env("AZURE_STORAGE_ACCOUNT", "sadronevideo"),
            storage_connection_string=_env("AZURE_STORAGE_CONNECTION_STRING"),
            account_key=_env("AZURE_ACCOUNT_KEY"),
            input_container=_env("AZURE_STORAGE_INPUT_CONTAINER", "input"),
            output_container=_env("AZURE_STORAGE_OUTPUT_CONTAINER", "output"),
            search_service_name=_env("AZURE_SEARCH_SERVICE_NAME"),
            search_endpoint=_env("AZURE_SEARCH_ENDPOINT"),
            search_admin_key=_env("AZURE_SEARCH_ADMIN_KEY"),
            search_index_name=_env("AZURE_SEARCH_INDEX_NAME", "dvsa-index"),
            search_sku=_env("AZURE_SEARCH_SKU", "standard"),
            vector_dimensions=int(_env("AZURE_SEARCH_VECTOR_DIMENSIONS", "1536")),
            openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=_env("AZURE_OPENAI_API_KEY"),
            openai_account=_env("AZURE_OPENAI_ACCOUNT", "dvsa-foundry"),
            gpt_deployment=_env("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-mini"),
            gpt_model=_env("AZURE_OPENAI_GPT_MODEL", "gpt-4o-mini"),
            embedding_deployment=_env(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
            ),
            embedding_model=_env(
                "AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"
            ),
            openai_api_version=_env("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            session_isolation=_env("AZURE_SESSION_ISOLATION", "index"),
            vision_api_version=_env("AZURE_VISION_API_VERSION", "2024-02-01"),
            search_api_version=_env("AZURE_SEARCH_API_VERSION", "2025-08-01-preview"),
            vision_model_version=_env("AZURE_VISION_MODEL_VERSION", "2023-04-15"),
            project_endpoint=_env("AZURE_PROJECT_ENDPOINT"),
            project_api_key=_env("AZURE_PROJECT_API_KEY"),
            agent_model=_env("AZURE_AGENT_MODEL", "gpt-4o-mini"),
            fn_agent_name=_env("AZURE_FN_AGENT_NAME", "fn-agent-in-a-team"),
            chat_agent_name=_env("AZURE_CHAT_AGENT_NAME", "chat-agent-in-a-team"),
            search_agent_name=_env("AZURE_SEARCH_AGENT_NAME", "search-agent-in-a-team"),
            tool_agent_name=_env("AZURE_TOOL_AGENT_NAME", "tool-agent-in-a-team"),
            search_connection_id=_env("AZURE_SEARCH_CONNECTION_ID"),
            search_resource_id=_env("AZURE_SEARCH_RESOURCE_ID"),
            search_subscription=_env("AZURE_SEARCH_SUBSCRIPTION"),
            search_resource_group=_env("AZURE_SEARCH_RESOURCE_GROUP"),
            search_location=_env("AZURE_SEARCH_LOCATION", "eastus"),
            video_indexer_url=_env(
                "AZURE_VIDEO_INDEXER_URL", "https://api.videoindexer.ai"
            ),
            video_indexer_region=_env("AZURE_VIDEO_INDEXER_REGION", "eastus"),
            video_indexer_account=_env("AZURE_VIDEO_INDEXER_ACCOUNT"),
            video_indexer_api_key=_env("AZURE_VIDEO_INDEXER_API_KEY"),
            video_indexer_access_token=_env("AZURE_VIDEO_INDEXER_ACCESS_TOKEN", ""),
            perplexity_chat_api_key=_env("PERPLEXITY_CHAT_API_KEY"),
            perplexity_chat_api_url=_env(
                "PERPLEXITY_CHAT_API_URL", "https://api.perplexity.ai/chat/completions"
            ),
            perplexity_geo_api_key=_env("PERPLEXITY_GEO_API_KEY"),
            perplexity_geo_api_url=_env(
                "PERPLEXITY_GEO_API_URL",
                "https://api.perplexity.ai/v1/image/geolocation",
            ),
            sample_object_uri=_env("SAMPLE_OBJECT_URI", ""),
            sample_scene_uri=_env("SAMPLE_SCENE_URI", ""),
        )

    # ----- helpers -------------------------------------------------------
    def control_plane_ready(self) -> bool:
        """True when ARM provisioning is possible (subscription configured)."""
        return bool(self.subscription_id)

    def search_data_plane_ready(self) -> bool:
        """True when the search index can be created/queried over the wire."""
        return bool(self.search_endpoint and self.search_admin_key)

    def resolve_mode(self) -> str:
        """Resolve the effective provisioner backend.

        ``auto`` becomes ``sdk`` when a subscription id is present, else
        ``dryrun``. Explicit modes are returned unchanged.
        """
        mode = (self.provisioner or "auto").lower()
        if mode != "auto":
            return mode
        return "sdk" if self.control_plane_ready() else "dryrun"
