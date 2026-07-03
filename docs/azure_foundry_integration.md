# Azure Foundry Reasoning Integration

A pluggable Azure Foundry reasoning provider for dvsa-api. It dynamically
provisions **ephemeral** Foundry reasoning deployments per session, runs
inference through a contract-compliant adapter, enforces cost/quota/time limits,
and tears everything down cleanly — automatically or on demand.

Out of the box it runs **fully offline** (dry-run): no Azure credentials, no
Terraform binary and no network are required to create a session, run inference,
and tear it down. That is exactly how the tests and CI exercise it.

## Architecture

```
        ┌────────────────────────── dvsa-api (Django/DRF) ──────────────────────────┐
        │                                                                            │
 client │  POST /api/reasoning/foundry/session ─────────► AzureFoundrySessionManager │
 ─────► │  POST /api/reasoning/foundry/session/{id}/infer                            │
        │  GET  /api/reasoning/foundry/session/{id}          │ provisions            │
        │  POST /api/reasoning/foundry/session/{id}/teardown │ (terraform | sdk |    │
        │                    │                               │  dry-run)             │
        │                    ▼                               ▼                       │
        │            AzureFoundryAdapter ──────────► AzureSdkWrapper (Key Vault,     │
        │            (predict/health_check)           Identity, Resource Mgmt)       │
        └────────────────────────────────────────────────────────────────────────────┘
                              │ terraform apply/destroy
                              ▼
        infra/azure/foundry_module  ──►  Resource Group ┐ Foundry account+deployment,
                                                        │ Managed Identity, Key Vault
                                                        │ (RBAC), optional VNet, roles
                                                        ┘ (the whole RG is the TTL unit)
```

Key modules:

| Path | Role |
| --- | --- |
| `dvsa_api/reasoning/adapter_base.py` | The `ReasoningModelAdapter` contract + response schema helpers. |
| `dvsa_api/reasoning/azure_foundry_adapter.py` | `predict` / `health_check`; managed + direct modes; telemetry. |
| `dvsa_api/reasoning/azure_session_manager.py` | Session lifecycle: provision, cost/quota/TTL, heartbeat, idempotent teardown, sweep, CLI. |
| `dvsa_api/reasoning/azure_sdk_wrapper.py` | Offline-safe Key Vault / Identity / Resource-Management wrapper with retry/backoff. |
| `dvsa_api/api/reasoning_foundry_router.py` | DRF endpoints under `/api/reasoning/foundry/`. |
| `infra/azure/foundry_module/` | Terraform module for one ephemeral session. |

## Quick start (dev, offline)

```bash
# 1) Create a session (dry-run — no cloud):
python scripts/foundry_session_cli.py create --model o1-mini --dry-run
# -> prints {"session_id": "...", "endpoint_url": "...", "expires_at": "...", ...}

# 2) Run an inference call:
python scripts/foundry_session_cli.py infer --session <id> --query "accidents?" --tracks 2

# 3) Inspect / tear down:
python scripts/foundry_session_cli.py status   --session <id>
python scripts/foundry_session_cli.py teardown --session <id>
```

Over HTTP (Django dev server):

```bash
curl -X POST localhost:8000/api/reasoning/foundry/session \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"o1-mini","max_duration_minutes":15,"max_cost_usd":0.5}'

curl -X POST localhost:8000/api/reasoning/foundry/session/<id>/infer \
  -H 'Content-Type: application/json' \
  -d '{"query":"accidents?","tracks":[{"id":"a"},{"id":"b"}]}'
```

### Modes

* **Managed session mode** (default): the session manager provisions resources
  and returns an endpoint; `predict` targets that endpoint.
* **Direct endpoint mode**: set `AZURE_FOUNDRY_ENDPOINT` and `AZURE_FOUNDRY_KEY`
  (the spec's `AZURE_FOUNDY_*` spelling is also accepted) and the adapter calls
  that endpoint with no provisioning.

## Configuration (environment variables)

| Var | Default | Purpose |
| --- | --- | --- |
| `AZURE_SUBSCRIPTION_ID` | – | Enables real provisioning (SDK/Terraform). |
| `AZURE_LOCATION` | `eastus` | Region for session resources. |
| `AZURE_FOUNDRY_TERRAFORM_DIR` | `infra/azure/foundry_module` | Module dir for terraform mode. |
| `USE_SDK_PROVISION` | `false` | Provision via Azure SDK instead of Terraform. |
| `AZURE_KEY_VAULT_URL` | – | Vault for endpoint keys (offline uses in-memory). |
| `AZURE_FOUNDRY_OFFLINE` | – | Force offline/dry-run everywhere. |
| `AZURE_FOUNDRY_ENDPOINT` / `AZURE_FOUNDRY_KEY` | – | Direct-endpoint mode. |
| `FOUNDRY_DEFAULT_DURATION_MINUTES` | `30` | Session TTL default. |
| `FOUNDRY_DEFAULT_MAX_COST_USD` | `1.0` | Soft cost cap default. |
| `FOUNDRY_INACTIVITY_TIMEOUT_MINUTES` | `15` | Idle teardown threshold. |
| `FOUNDRY_MAX_ACTIVE_SESSIONS` / `_PER_USER` | `25` / `5` | Quotas. |
| `FOUNDRY_SOFT_STOP_FRACTION` | `0.8` | Fraction of cap that triggers a soft stop. |
| `AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_CLIENT_SECRET` | – | Honoured by `DefaultAzureCredential` for CI/local. |

## Security

* **Managed identity** — the Foundry deployment runs as a user-assigned managed
  identity granted only **Key Vault Secrets User** (read) on the session vault.
* **Key Vault** — endpoint keys live in Key Vault; the app writes the secret
  (the Terraform module never stores secret values in state). Secrets are never
  logged and are stripped from telemetry / `to_public_dict()`.
* **RBAC vs access policies** — the module uses RBAC
  (`enable_rbac_authorization = true`). For tenants standardised on **access
  policies**, replace the two `azurerm_role_assignment` resources with
  `azurerm_key_vault_access_policy` blocks granting `secret_permissions = ["Get"]`
  to the identity and `["Set","Get","Delete"]` to the deployer.
* **Least privilege** — everything is scoped to a single resource group; the
  module creates no subscription-level resources.
* **Private networking** — set `private_networking = true` to provision a
  VNet/subnet and switch the vault + account to deny-by-default.

## Cost controls

* Conservative defaults: `max_duration_minutes=30`, `max_cost_usd=1.0`.
* An **up-front estimate** (`cost_estimate`) is returned at create time from a
  SKU→hourly-rate map; creation is rejected if it already exceeds the cap.
* **Soft stop**: at `FOUNDRY_SOFT_STOP_FRACTION` of the cap, further inference is
  blocked until the caller passes `force: true`.
* **Hard stop**: exceeding the cap tears the session down automatically
  (`status = over_budget`).
* **TTL & inactivity**: `sweep()` (wire to cron/Celery beat) reaps expired,
  inactive and over-budget sessions.
* **Fallback**: set `AZURE_FOUNDRY_FALLBACK` (or run in direct-endpoint mode) to
  route to a pre-provisioned shared endpoint for low-cost/emergency use.

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `provisioning_failed` | Terraform/SDK error. Check `AZURE_SUBSCRIPTION_ID` + creds; best-effort cleanup already ran, so no partial resources should remain. |
| `session_not_found` on infer | Session expired (TTL) or was torn down. Create a new one. |
| `cost_limit_exceeded` on infer | Soft stop hit — retry with `{"force": true}` or raise `max_cost_usd`. |
| `quota_exceeded` | Global/per-user active-session cap reached — tear down idle sessions. |
| `model_unavailable` | Endpoint unreachable, or neither a managed session nor a direct endpoint is configured. |
| Terraform `plan` fails in CI | Expected without live creds; CI gates on `fmt` + `validate` only. |

## Operational runbook

* **Manually tear down a stuck session**:
  `python scripts/foundry_session_cli.py teardown --session <id>`, or
  `terraform -chdir=infra/azure/foundry_module destroy -var-file=terraform.tfvars`.
  Deleting the session's **resource group** removes everything at once.
* **Rotate keys**: write a new value to the Key Vault secret named by
  `key_vault_secret_name`; the adapter reads it on the next call.
* **Audit usage**: `GET /api/reasoning/foundry/session/{id}` returns
  `usage.{calls,tokens,estimated_cost_usd}` and cost limits. Provisioning ops are
  recorded on `session.provision_ops`.
* **Reap everything**: call `AzureFoundrySessionManager.sweep()` on a schedule.

## Testing / CI

* Unit tests mock the Azure SDK and Terraform and run with zero network:
  `pytest tests/test_azure_session_manager.py tests/test_azure_foundry_adapter.py tests/test_foundry_router.py`.
* `scripts/terraform_local_plan.sh` runs `terraform fmt`/`validate` (enforced)
  and a best-effort plan-only pass. It never applies.
* `.github/workflows/ci-foundry.yml` wires both together on PRs and pushes.
