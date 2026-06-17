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

    # Agent names ported from ezvision settings.py (Foundry agent roles)
    agent_names: tuple = field(
        default=(
            "fn-agent-in-a-team",
            "chat-agent-in-a-team",
            "search-agent-in-a-team",
            "tool-agent-in-a-team",
        )
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
