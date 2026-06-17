"""Session-scoped Azure environment for DVSA.

This package ports the Azure Resource Manager (ARM) usage that lived in
``ezvision``'s ``my_droneworld_api/settings.py`` into a cohesive, reusable
setup/teardown API for ``dvsa-api``.

Public surface
--------------
- :class:`~core.azure.config.AzureEnvironmentConfig` — the configuration
  contract (env/Django-settings driven).
- :func:`~core.azure.session.create_session_azure_environment` — provision (or
  attach to) the Azure resources a single user session needs.
- :func:`~core.azure.session.teardown_session_azure_environment` — release /
  delete the session's resources.
- :class:`~core.azure.session.SessionAzureEnvironment` — the handle returned by
  ``create_...``; carries ready-to-use data-plane clients and config.

Design notes
------------
*Everything stays importable and exercisable without the Azure SDKs or any
credentials.* When credentials are absent (or ``AZURE_PROVISIONER=dryrun``) the
provisioner runs in **dry-run** mode and records the operations it *would* have
performed — mirroring the offline ``EchoLLMClient`` pattern already used in
``apps.observability.llm``. Real SDK packages are imported lazily so a missing
``azure-mgmt-*`` wheel never breaks ``import core.azure``.
"""

from .config import AzureEnvironmentConfig
from .session import (
    SessionAzureEnvironment,
    create_session_azure_environment,
    teardown_session_azure_environment,
)


def get_vision_client(config=None):
    """Convenience factory for the Vision data-plane client."""
    from .vision import VisionClient

    return VisionClient(config or AzureEnvironmentConfig.from_settings())


def get_video_indexer_client(config=None):
    """Convenience factory for the Video Indexer client."""
    from .video_indexer import VideoIndexerClient

    return VideoIndexerClient(config or AzureEnvironmentConfig.from_settings())


def get_foundry_agents(config=None):
    """Convenience factory for the Foundry agents runtime."""
    from .agents import FoundryAgents

    return FoundryAgents(config or AzureEnvironmentConfig.from_settings())


__all__ = [
    "AzureEnvironmentConfig",
    "SessionAzureEnvironment",
    "create_session_azure_environment",
    "teardown_session_azure_environment",
    "get_vision_client",
    "get_video_indexer_client",
    "get_foundry_agents",
]
