#!/bin/bash
# Cluster-scoped init script: install the DVSA Databricks connector on startup.
#
# Attach this under Compute > <cluster> > Advanced options > Init Scripts, or
# reference it from a cluster policy. It installs the connector wheel (and,
# optionally, a private in-cluster DVSA adapter) onto every node so notebooks
# and Jobs can `import dvsa_databricks_connector` without a %pip cell.
#
# Two install sources are supported; set ONE of these as a cluster env var:
#   * DVSA_CONNECTOR_WHEEL  — dbfs:/ or /Volumes path to a pre-built wheel
#   * DVSA_CONNECTOR_GIT    — pip VCS URL, e.g.
#       git+https://github.com/ravibeta/DVSA-APIs#subdirectory=dvsa_databricks_connector
#
# Never put the DVSA API key here — store it in Databricks Secrets and read it
# at runtime via dvsa_databricks_connector.config_from_secrets(dbutils).
set -euo pipefail

echo "[dvsa-init] installing DVSA Databricks connector..."

if [[ -n "${DVSA_CONNECTOR_WHEEL:-}" ]]; then
  echo "[dvsa-init] installing from wheel: ${DVSA_CONNECTOR_WHEEL}"
  /databricks/python/bin/pip install --upgrade "${DVSA_CONNECTOR_WHEEL}"
elif [[ -n "${DVSA_CONNECTOR_GIT:-}" ]]; then
  echo "[dvsa-init] installing from git: ${DVSA_CONNECTOR_GIT}"
  /databricks/python/bin/pip install --upgrade "${DVSA_CONNECTOR_GIT}"
else
  echo "[dvsa-init] ERROR: set DVSA_CONNECTOR_WHEEL or DVSA_CONNECTOR_GIT" >&2
  exit 1
fi

# Optional: install a private in-cluster DVSA adapter for in_cluster mode.
if [[ -n "${DVSA_ADAPTER_WHEEL:-}" ]]; then
  echo "[dvsa-init] installing in-cluster DVSA adapter: ${DVSA_ADAPTER_WHEEL}"
  /databricks/python/bin/pip install --upgrade "${DVSA_ADAPTER_WHEEL}"
fi

echo "[dvsa-init] done."
