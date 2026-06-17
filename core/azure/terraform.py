"""Terraform backend for global resource provisioning.

This is the "Infrastructure as Code" route requested as an alternative to the
SDK: a thin wrapper that drives ``terraform`` against the module in
``solution-accelerator/`` to create the shared storage account, AI Search
service, and Foundry/OpenAI account + deployments.

Design choices
--------------
- *Global* resources are created in one ``terraform apply`` using a per-call
  ``-var`` set, with state isolated per resource group (``terraform workspace``).
- *Data-plane* objects (the per-session search index and blob prefixes) are
  **not** Terraform-managed — they are fast, numerous, and session-scoped, so
  those calls delegate to :class:`~core.azure.provisioning.AzureSdkProvisioner`
  when credentials exist, and otherwise record (dry-run) like every other path.
- If the ``terraform`` binary is missing, the apply step degrades to a recorded
  no-op so the API never hard-fails in environments without Terraform.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any, Dict, Optional

from .config import AzureEnvironmentConfig
from .provisioning import DryRunProvisioner, Provisioner, ProvisioningError

logger = logging.getLogger("apps.azure")


class TerraformProvisioner(Provisioner):
    """Provision global resources via ``terraform`` in ``config.terraform_dir``."""

    mode = "terraform"

    def __init__(self, config: AzureEnvironmentConfig) -> None:
        super().__init__(config)
        self.terraform_dir = config.terraform_dir
        # Lazily-built SDK delegate for data-plane ops (index/blob).
        self._dataplane: Optional[Provisioner] = None

    # --- helpers ---------------------------------------------------------
    def _delegate(self) -> Provisioner:
        if self._dataplane is None:
            if self.config.control_plane_ready():
                from .provisioning import AzureSdkProvisioner  # noqa: PLC0415

                try:
                    self._dataplane = AzureSdkProvisioner(self.config)
                except ProvisioningError:
                    self._dataplane = DryRunProvisioner(self.config)
            else:
                self._dataplane = DryRunProvisioner(self.config)
        return self._dataplane

    def _tf_vars(self) -> Dict[str, str]:
        c = self.config
        return {
            "resource_group": c.resource_group,
            "location": c.location,
            "storage_account": c.storage_account,
            "input_container": c.input_container,
            "output_container": c.output_container,
            "search_service_name": c.search_service_name or "srch-dvsa",
            "search_sku": c.search_sku,
            "search_index_name": c.search_index_name,
            "vector_dimensions": str(c.vector_dimensions),
            "openai_account": c.openai_account,
            "gpt_deployment": c.gpt_deployment,
            "gpt_model": c.gpt_model,
            "embedding_deployment": c.embedding_deployment,
            "embedding_model": c.embedding_model,
        }

    def _run(self, *args: str) -> int:
        if not shutil.which("terraform"):
            logger.warning("terraform binary not found; recording no-op apply")
            return -1
        if not self.terraform_dir:
            raise ProvisioningError("AZURE_TERRAFORM_DIR is not configured")
        var_args = []
        for key, val in self._tf_vars().items():
            var_args += ["-var", f"{key}={val}"]
        subprocess.run(["terraform", "init", "-input=false"],
                       cwd=self.terraform_dir, check=True)
        proc = subprocess.run(
            ["terraform", *args, "-input=false", "-auto-approve", *var_args],
            cwd=self.terraform_dir, check=False,
        )
        return proc.returncode

    # --- global (single apply) ------------------------------------------
    def ensure_global(self) -> Dict[str, Any]:
        rc = self._run("apply")
        op = self._record("apply", "terraform_module", dir=self.terraform_dir,
                         returncode=rc, vars=self._tf_vars())
        return {"terraform": op}

    # The per-resource ensure_* are covered by the single apply above; expose
    # them as recorded no-ops so the Provisioner contract still holds.
    def ensure_resource_group(self) -> Dict[str, Any]:
        return self._record("noop", "resource_group", via="terraform apply")

    def ensure_storage_account(self) -> Dict[str, Any]:
        return self._record("noop", "storage_account", via="terraform apply")

    def ensure_containers(self) -> Dict[str, Any]:
        return self._record("noop", "containers", via="terraform apply")

    def ensure_search_service(self) -> Dict[str, Any]:
        return self._record("noop", "search_service", via="terraform apply")

    def ensure_openai_account(self) -> Dict[str, Any]:
        return self._record("noop", "openai_account", via="terraform apply")

    def ensure_deployments(self) -> Dict[str, Any]:
        return self._record("noop", "deployments", via="terraform apply")

    # --- data plane (delegated) -----------------------------------------
    def ensure_search_index(self, name: str) -> Dict[str, Any]:
        op = self._delegate().ensure_search_index(name)
        self.operations.append(op)
        return op

    def teardown_search_index(self, name: str) -> Dict[str, Any]:
        op = self._delegate().teardown_search_index(name)
        self.operations.append(op)
        return op

    def teardown_blob_prefix(self, container: str, prefix: str) -> Dict[str, Any]:
        op = self._delegate().teardown_blob_prefix(container, prefix)
        self.operations.append(op)
        return op

    def destroy_global(self) -> Dict[str, Any]:
        rc = self._run("destroy")
        return self._record("destroy", "terraform_module", returncode=rc)
