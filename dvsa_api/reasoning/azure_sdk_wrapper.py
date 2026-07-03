"""Thin, offline-safe wrapper around the Azure SDKs used by the adapter.

Only three capabilities are needed by the Foundry integration:

* **credentials** — :class:`azure.identity.DefaultAzureCredential`, with the
  ``AZURE_CLIENT_ID`` / ``AZURE_TENANT_ID`` / ``AZURE_CLIENT_SECRET`` triple
  honoured automatically by ``DefaultAzureCredential`` for CI/local dev.
* **secrets** — read/write/delete endpoint keys in Key Vault
  (:mod:`azure.keyvault.secrets`).
* **resources** — resource-group create/delete when ``USE_SDK_PROVISION=true``
  (:mod:`azure.mgmt.resource`).

Design mirrors ``core/azure/provisioning.py``: SDKs are imported lazily and, when
credentials or packages are missing, the wrapper degrades to an **in-memory
store** so the whole stack runs and is testable offline with zero network calls.
Every public method is easily monkeypatched in tests.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from .errors import AuthError, ProvisioningError, TeardownError

logger = logging.getLogger("dvsa_api.reasoning")

T = TypeVar("T")

# Value stored in place of a real secret while offline, so nothing is ever
# mistaken for a live key and logs stay clean.
_OFFLINE_SECRET_SENTINEL = "offline-dryrun-secret"


def _truthy(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.2,
    exc_type: type = Exception,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``fn`` with exponential backoff, re-raising the last error.

    ``sleep`` is injectable so tests can run instantly.
    """
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except exc_type as exc:  # noqa: BLE001 - deliberately broad, re-raised
            last = exc
            if i == attempts - 1:
                break
            delay = base_delay * (2 ** i)
            logger.warning("azure call failed (attempt %d/%d): %s; retrying in %.2fs",
                           i + 1, attempts, exc, delay)
            sleep(delay)
    assert last is not None  # for type checkers; loop always sets it on failure
    raise last


class AzureSdkWrapper:
    """Offline-safe facade over azure-identity / keyvault / mgmt-resource.

    Parameters
    ----------
    key_vault_url:
        ``https://<vault>.vault.azure.net``. When falsy (or SDKs/credentials are
        unavailable) all secret ops use the in-memory store.
    subscription_id:
        Needed for resource-group operations in SDK provisioning mode.
    online:
        Force online/offline. ``None`` (default) auto-detects: online only when a
        vault URL or subscription id is present *and* ``AZURE_FOUNDRY_OFFLINE``
        is not set.
    """

    def __init__(
        self,
        *,
        key_vault_url: Optional[str] = None,
        subscription_id: Optional[str] = None,
        online: Optional[bool] = None,
    ) -> None:
        self.key_vault_url = key_vault_url or os.environ.get("AZURE_KEY_VAULT_URL")
        self.subscription_id = subscription_id or os.environ.get(
            "AZURE_SUBSCRIPTION_ID"
        )
        if online is None:
            forced_offline = _truthy("AZURE_FOUNDRY_OFFLINE")
            online = bool(self.key_vault_url or self.subscription_id) and not forced_offline
        self.online = bool(online)
        self._cred = None
        self._secret_client = None
        # In-memory secret store used offline (and as a write-through cache).
        self._mem_secrets: Dict[str, str] = {}

    # ----- credentials ---------------------------------------------------
    @property
    def credential(self):
        """Lazily build a ``DefaultAzureCredential`` (raises AuthError on failure)."""
        if self._cred is None:
            try:
                from azure.identity import DefaultAzureCredential  # noqa: PLC0415

                self._cred = DefaultAzureCredential()
            except Exception as exc:  # noqa: BLE001
                raise AuthError(f"could not build Azure credential: {exc}") from exc
        return self._cred

    # ----- key vault secrets --------------------------------------------
    def _secrets(self):
        if self._secret_client is None:
            from azure.keyvault.secrets import SecretClient  # noqa: PLC0415

            self._secret_client = SecretClient(
                vault_url=self.key_vault_url, credential=self.credential
            )
        return self._secret_client

    def set_secret(self, name: str, value: str) -> str:
        """Store ``value`` under ``name``; returns the secret name.

        Offline (or without a vault) the value is kept in memory only.
        """
        self._mem_secrets[name] = value
        if not (self.online and self.key_vault_url):
            logger.info("offline set_secret %s (in-memory)", name)
            return name
        try:
            retry(lambda: self._secrets().set_secret(name, value))
        except Exception as exc:  # noqa: BLE001
            raise ProvisioningError(
                f"failed to store secret '{name}'", details={"error": str(exc)}
            ) from exc
        return name

    def get_secret(self, name: str) -> str:
        """Return the secret value for ``name`` (never logged)."""
        if not (self.online and self.key_vault_url):
            return self._mem_secrets.get(name, _OFFLINE_SECRET_SENTINEL)
        try:
            return retry(lambda: self._secrets().get_secret(name).value)
        except Exception as exc:  # noqa: BLE001
            # Fall back to any write-through cached copy before giving up.
            if name in self._mem_secrets:
                return self._mem_secrets[name]
            raise ProvisioningError(
                f"failed to read secret '{name}'", details={"error": str(exc)}
            ) from exc

    def delete_secret(self, name: str) -> None:
        """Best-effort secret removal; missing secrets are not an error."""
        self._mem_secrets.pop(name, None)
        if not (self.online and self.key_vault_url):
            return
        try:
            retry(lambda: self._secrets().begin_delete_secret(name))
        except Exception as exc:  # noqa: BLE001 - deletion is best-effort
            logger.warning("delete_secret %s failed (ignored): %s", name, exc)

    # ----- resource groups (SDK provisioning mode) -----------------------
    def _resource_client(self):
        if not self.subscription_id:
            raise ProvisioningError("AZURE_SUBSCRIPTION_ID required for SDK provisioning")
        from azure.mgmt.resource import ResourceManagementClient  # noqa: PLC0415

        return ResourceManagementClient(self.credential, self.subscription_id)

    def ensure_resource_group(self, name: str, location: str) -> Dict[str, Any]:
        """Create/update a resource group; offline returns a synthetic record."""
        if not self.online:
            logger.info("offline ensure_resource_group %s", name)
            return {"name": name, "location": location, "provisioned": False}
        try:
            retry(
                lambda: self._resource_client().resource_groups.create_or_update(
                    name, {"location": location}
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise ProvisioningError(
                f"resource group '{name}' failed", details={"error": str(exc)}
            ) from exc
        return {"name": name, "location": location, "provisioned": True}

    def delete_resource_group(self, name: str) -> Dict[str, Any]:
        """Delete a resource group; idempotent and offline-safe."""
        if not self.online:
            logger.info("offline delete_resource_group %s", name)
            return {"name": name, "deleted": False}
        try:
            retry(lambda: self._resource_client().resource_groups.begin_delete(name))
        except Exception as exc:  # noqa: BLE001
            raise TeardownError(
                f"resource group '{name}' delete failed", details={"error": str(exc)}
            ) from exc
        return {"name": name, "deleted": True}
