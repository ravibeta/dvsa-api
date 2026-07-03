"""Pluggable reasoning-model runtime + the Azure Foundry provider.

Public surface
--------------
* :class:`ReasoningModelAdapter` — the adapter contract every provider implements.
* registry helpers — :func:`get_model`, :func:`list_models`, :func:`call_model`.
* :class:`AzureFoundryAdapter` — the Azure Foundry reasoning adapter.
* :class:`AzureFoundrySessionManager` / :func:`get_session_manager` — ephemeral
  Foundry session lifecycle.

Importing this package registers the Foundry provider under ``"azure_foundry"``
so it participates in the shared registry/fallback machinery.
"""

from __future__ import annotations

from .adapter_base import ReasoningModelAdapter
from .azure_foundry_adapter import AzureFoundryAdapter, build_adapter_for_session
from .azure_session_manager import (
    AzureFoundrySessionManager,
    FoundrySession,
    get_session_manager,
    reset_session_manager,
)
from .errors import (
    CostLimitExceeded,
    ModelUnavailableError,
    ProvisioningError,
    QuotaExceeded,
    SessionNotFoundError,
    TeardownError,
)
from .registry import call_model, get_model, list_models, register, register_instance

# Register the Foundry provider factory. A no-arg AzureFoundryAdapter uses
# direct-endpoint mode when AZURE_FOUNDRY_ENDPOINT is set; managed sessions bind
# their own adapter via build_adapter_for_session().
register("azure_foundry", AzureFoundryAdapter)

__all__ = [
    "ReasoningModelAdapter",
    "AzureFoundryAdapter",
    "build_adapter_for_session",
    "AzureFoundrySessionManager",
    "FoundrySession",
    "get_session_manager",
    "reset_session_manager",
    "call_model",
    "get_model",
    "list_models",
    "register",
    "register_instance",
    "ProvisioningError",
    "TeardownError",
    "SessionNotFoundError",
    "ModelUnavailableError",
    "CostLimitExceeded",
    "QuotaExceeded",
]
