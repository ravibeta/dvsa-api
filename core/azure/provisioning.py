"""Idempotent provisioning backends for the Azure environment.

A :class:`Provisioner` exposes ``ensure_*`` (create-if-absent) and ``teardown_*``
operations. Every call is idempotent and records a structured op into
``self.operations`` for logging/auditing.

Backends
--------
- :class:`DryRunProvisioner` — no network, no credentials. Records the
  operations it *would* run and returns synthetic endpoints. Default whenever a
  subscription id is missing, so the whole stack is exercisable offline (same
  philosophy as ``apps.observability.llm.EchoLLMClient``).
- :class:`AzureSdkProvisioner` — real ARM via ``azure-identity`` +
  ``azure-mgmt-*`` and data-plane via ``azure-storage-blob`` /
  ``azure-search-documents``. SDKs are imported lazily.

Use :func:`get_provisioner` to pick a backend from config.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional

from .config import AzureEnvironmentConfig
from . import index_schema

logger = logging.getLogger("apps.azure")


class ProvisioningError(RuntimeError):
    """Raised when a provisioning step fails irrecoverably."""


class Provisioner(abc.ABC):
    """Idempotent setup/teardown of Azure resources for the environment."""

    mode = "base"

    def __init__(self, config: AzureEnvironmentConfig) -> None:
        self.config = config
        self.operations: List[Dict[str, Any]] = []

    def _record(self, action: str, resource: str, **detail: Any) -> Dict[str, Any]:
        op = {"mode": self.mode, "action": action, "resource": resource, **detail}
        self.operations.append(op)
        logger.info("azure.%s %s %s", self.mode, action, resource)
        return op

    # ----- global (control-plane) resources ------------------------------
    @abc.abstractmethod
    def ensure_resource_group(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def ensure_storage_account(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def ensure_containers(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def ensure_search_service(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def ensure_openai_account(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def ensure_deployments(self) -> Dict[str, Any]: ...

    # ----- session (data-plane) resources --------------------------------
    @abc.abstractmethod
    def ensure_search_index(self, name: str) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def teardown_search_index(self, name: str) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def teardown_blob_prefix(self, container: str, prefix: str) -> Dict[str, Any]: ...

    # ----- orchestration -------------------------------------------------
    def ensure_global(self) -> Dict[str, Any]:
        """Ensure all shared/global resources exist (idempotent)."""
        return {
            "resource_group": self.ensure_resource_group(),
            "storage_account": self.ensure_storage_account(),
            "containers": self.ensure_containers(),
            "search_service": self.ensure_search_service(),
            "openai_account": self.ensure_openai_account(),
            "deployments": self.ensure_deployments(),
        }


class DryRunProvisioner(Provisioner):
    """Offline provisioner — records intended operations, performs no I/O."""

    mode = "dryrun"

    def ensure_resource_group(self) -> Dict[str, Any]:
        return self._record("ensure", "resource_group",
                            name=self.config.resource_group,
                            location=self.config.location)

    def ensure_storage_account(self) -> Dict[str, Any]:
        acct = self.config.storage_account
        return self._record(
            "ensure", "storage_account", name=acct,
            endpoint=f"https://{acct}.blob.core.windows.net",
        )

    def ensure_containers(self) -> Dict[str, Any]:
        names = [self.config.input_container, self.config.output_container]
        return self._record("ensure", "containers", names=names)

    def ensure_search_service(self) -> Dict[str, Any]:
        name = self.config.search_service_name or "srch-dvsa"
        return self._record(
            "ensure", "search_service", name=name, sku=self.config.search_sku,
            endpoint=self.config.search_endpoint
            or f"https://{name}.search.windows.net",
        )

    def ensure_openai_account(self) -> Dict[str, Any]:
        acct = self.config.openai_account
        return self._record(
            "ensure", "openai_account", name=acct,
            endpoint=self.config.openai_endpoint
            or f"https://{acct}.openai.azure.com",
        )

    def ensure_deployments(self) -> Dict[str, Any]:
        deployments = [
            {"name": self.config.embedding_deployment,
             "model": self.config.embedding_model, "role": "embedding"},
            {"name": self.config.gpt_deployment,
             "model": self.config.gpt_model, "role": "chat"},
        ]
        return self._record("ensure", "deployments", deployments=deployments,
                            agents=list(self.config.agent_names))

    def ensure_search_index(self, name: str) -> Dict[str, Any]:
        desc = index_schema.describe_index(name, self.config.vector_dimensions)
        return self._record("ensure", "search_index", name=name,
                            dimensions=self.config.vector_dimensions,
                            fields=[f["name"] for f in desc["fields"]])

    def teardown_search_index(self, name: str) -> Dict[str, Any]:
        return self._record("delete", "search_index", name=name)

    def teardown_blob_prefix(self, container: str, prefix: str) -> Dict[str, Any]:
        return self._record("delete", "blob_prefix", container=container,
                            prefix=prefix)


class AzureSdkProvisioner(Provisioner):
    """Real provisioner using azure-identity + azure-mgmt-* + data-plane SDKs."""

    mode = "sdk"

    def __init__(self, config: AzureEnvironmentConfig) -> None:
        super().__init__(config)
        if not config.control_plane_ready():
            raise ProvisioningError(
                "AzureSdkProvisioner requires AZURE_SUBSCRIPTION_ID"
            )
        self._cred = None
        self._account_key: Optional[str] = config.account_key

    # --- lazy clients ----------------------------------------------------
    @property
    def credential(self):
        if self._cred is None:
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415

            self._cred = DefaultAzureCredential()
        return self._cred

    def _resource_client(self):
        from azure.mgmt.resource import ResourceManagementClient  # noqa: PLC0415

        return ResourceManagementClient(self.credential, self.config.subscription_id)

    def _storage_mgmt(self):
        from azure.mgmt.storage import StorageManagementClient  # noqa: PLC0415

        return StorageManagementClient(self.credential, self.config.subscription_id)

    def _search_mgmt(self):
        from azure.mgmt.search import SearchManagementClient  # noqa: PLC0415

        return SearchManagementClient(self.credential, self.config.subscription_id)

    def _cognitive_mgmt(self):
        from azure.mgmt.cognitiveservices import (  # noqa: PLC0415
            CognitiveServicesManagementClient,
        )

        return CognitiveServicesManagementClient(
            self.credential, self.config.subscription_id
        )

    # --- global resources ------------------------------------------------
    def ensure_resource_group(self) -> Dict[str, Any]:
        c = self.config
        try:
            self._resource_client().resource_groups.create_or_update(
                c.resource_group, {"location": c.location}
            )
        except Exception as exc:  # noqa: BLE001
            raise ProvisioningError(f"resource group failed: {exc}") from exc
        return self._record("ensure", "resource_group", name=c.resource_group,
                            location=c.location)

    def ensure_storage_account(self) -> Dict[str, Any]:
        c = self.config
        client = self._storage_mgmt()
        try:
            existing = list(
                client.storage_accounts.list_by_resource_group(c.resource_group)
            )
            if not any(a.name == c.storage_account for a in existing):
                poller = client.storage_accounts.begin_create(
                    c.resource_group, c.storage_account,
                    {
                        "sku": {"name": "Standard_LRS"},
                        "kind": "StorageV2",
                        "location": c.location,
                    },
                )
                poller.result()
            keys = client.storage_accounts.list_keys(
                c.resource_group, c.storage_account
            )
            self._account_key = keys.keys[0].value
        except Exception as exc:  # noqa: BLE001
            raise ProvisioningError(f"storage account failed: {exc}") from exc
        return self._record(
            "ensure", "storage_account", name=c.storage_account,
            endpoint=f"https://{c.storage_account}.blob.core.windows.net",
        )

    def ensure_containers(self) -> Dict[str, Any]:
        c = self.config
        from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

        svc = BlobServiceClient(
            account_url=f"https://{c.storage_account}.blob.core.windows.net",
            credential=self._account_key or self.credential,
        )
        created = []
        for name in (c.input_container, c.output_container):
            try:
                svc.create_container(name)
                created.append(name)
            except Exception as exc:  # noqa: BLE001 - already-exists is fine
                if "ContainerAlreadyExists" not in str(exc):
                    raise ProvisioningError(f"container {name} failed: {exc}") from exc
        return self._record("ensure", "containers",
                            names=[c.input_container, c.output_container],
                            created=created)

    def ensure_search_service(self) -> Dict[str, Any]:
        c = self.config
        name = c.search_service_name
        if not name:
            raise ProvisioningError("AZURE_SEARCH_SERVICE_NAME is required for SDK mode")
        client = self._search_mgmt()
        try:
            poller = client.services.begin_create_or_update(
                c.resource_group, name,
                {"location": c.location, "sku": {"name": c.search_sku},
                 "replica_count": 1, "partition_count": 1},
            )
            poller.result()
            keys = client.admin_keys.get(c.resource_group, name)
            admin_key = keys.primary_key
        except Exception as exc:  # noqa: BLE001
            raise ProvisioningError(f"search service failed: {exc}") from exc
        return self._record(
            "ensure", "search_service", name=name, sku=c.search_sku,
            endpoint=f"https://{name}.search.windows.net", admin_key=admin_key,
        )

    def ensure_openai_account(self) -> Dict[str, Any]:
        c = self.config
        client = self._cognitive_mgmt()
        try:
            poller = client.accounts.begin_create(
                c.resource_group, c.openai_account,
                {
                    "location": c.location,
                    "kind": "OpenAI",
                    "sku": {"name": "S0"},
                    "properties": {"custom_sub_domain_name": c.openai_account},
                },
            )
            poller.result()
            keys = client.accounts.list_keys(c.resource_group, c.openai_account)
            api_key = keys.key1
        except Exception as exc:  # noqa: BLE001
            raise ProvisioningError(f"openai account failed: {exc}") from exc
        return self._record(
            "ensure", "openai_account", name=c.openai_account,
            endpoint=c.openai_endpoint or f"https://{c.openai_account}.openai.azure.com",
            api_key=api_key,
        )

    def ensure_deployments(self) -> Dict[str, Any]:
        c = self.config
        client = self._cognitive_mgmt()
        targets = [
            (c.embedding_deployment, c.embedding_model, "embedding"),
            (c.gpt_deployment, c.gpt_model, "chat"),
        ]
        done = []
        for dep_name, model, role in targets:
            try:
                poller = client.deployments.begin_create_or_update(
                    c.resource_group, c.openai_account, dep_name,
                    {"properties": {"model": {"format": "OpenAI", "name": model}},
                     "sku": {"name": "Standard", "capacity": 1}},
                )
                poller.result()
                done.append({"name": dep_name, "model": model, "role": role})
            except Exception as exc:  # noqa: BLE001
                raise ProvisioningError(
                    f"deployment {dep_name} failed: {exc}"
                ) from exc
        return self._record("ensure", "deployments", deployments=done,
                            agents=list(c.agent_names))

    # --- data plane ------------------------------------------------------
    def _index_client(self):
        from azure.core.credentials import AzureKeyCredential  # noqa: PLC0415
        from azure.search.documents.indexes import (  # noqa: PLC0415
            SearchIndexClient,
        )

        endpoint = self.config.search_endpoint
        key = self.config.search_admin_key
        if not (endpoint and key):
            raise ProvisioningError(
                "search index ops need AZURE_SEARCH_ENDPOINT + AZURE_SEARCH_ADMIN_KEY"
            )
        return SearchIndexClient(endpoint, AzureKeyCredential(key))

    def ensure_search_index(self, name: str) -> Dict[str, Any]:
        index = index_schema.build_search_index(name, self.config.vector_dimensions)
        try:
            self._index_client().create_or_update_index(index)
        except Exception as exc:  # noqa: BLE001
            raise ProvisioningError(f"search index {name} failed: {exc}") from exc
        return self._record("ensure", "search_index", name=name,
                            dimensions=self.config.vector_dimensions)

    def teardown_search_index(self, name: str) -> Dict[str, Any]:
        try:
            self._index_client().delete_index(name)
        except Exception as exc:  # noqa: BLE001 - missing index is fine
            logger.warning("delete index %s: %s", name, exc)
        return self._record("delete", "search_index", name=name)

    def teardown_blob_prefix(self, container: str, prefix: str) -> Dict[str, Any]:
        from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

        c = self.config
        svc = BlobServiceClient(
            account_url=f"https://{c.storage_account}.blob.core.windows.net",
            credential=self._account_key or self.credential,
        )
        deleted = 0
        try:
            container_client = svc.get_container_client(container)
            for blob in container_client.list_blobs(name_starts_with=prefix):
                container_client.delete_blob(blob.name)
                deleted += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete prefix %s/%s: %s", container, prefix, exc)
        return self._record("delete", "blob_prefix", container=container,
                            prefix=prefix, deleted=deleted)


def get_provisioner(
    config: AzureEnvironmentConfig, mode: Optional[str] = None
) -> Provisioner:
    """Return a provisioner for ``mode`` (defaults to ``config.resolve_mode()``).

    ``terraform`` is delegated to :class:`core.azure.terraform.TerraformProvisioner`.
    Any unknown / SDK-unavailable mode degrades to dry-run with a warning.
    """
    mode = (mode or config.resolve_mode()).lower()
    if mode == "dryrun":
        return DryRunProvisioner(config)
    if mode == "terraform":
        from .terraform import TerraformProvisioner  # noqa: PLC0415

        return TerraformProvisioner(config)
    if mode == "sdk":
        try:
            return AzureSdkProvisioner(config)
        except ProvisioningError as exc:
            logger.warning("falling back to dry-run: %s", exc)
            return DryRunProvisioner(config)
    logger.warning("unknown provisioner '%s'; using dry-run", mode)
    return DryRunProvisioner(config)
