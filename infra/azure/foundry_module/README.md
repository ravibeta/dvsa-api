# `foundry_module` — ephemeral Azure Foundry reasoning session

This Terraform module provisions, **inside a single resource group**, everything
one ephemeral Foundry reasoning session needs:

| Resource | Purpose |
| --- | --- |
| `azurerm_resource_group` | The teardown unit — destroy it and the session is gone. |
| `azurerm_cognitive_account` (`kind = OpenAI`) + `azurerm_cognitive_deployment` | The Foundry reasoning model deployment. |
| `azurerm_user_assigned_identity` | Least-privilege identity used by the deployment. |
| `azurerm_key_vault` (RBAC-authorized) | Stores the endpoint key. |
| `azurerm_role_assignment` ×2 | Identity gets **Secrets User** (read); deployer gets **Secrets Officer** (write once). |
| `azurerm_virtual_network` / `azurerm_subnet` | Optional, only when `private_networking = true`. |

The module never creates subscription-level resources.

## Variables

See [`variables.tf`](./variables.tf). Required: `resource_group_name`,
`session_name`, `managed_identity_name`. Everything else has safe defaults.

## Outputs

`foundry_endpoint_url`, `foundry_deployment_id`, `managed_identity_principal_id`,
`key_vault_name`, `key_vault_secret_name`, `resource_group_name`.

## Plan-only in CI (no live Azure)

`terraform validate` and `terraform fmt` need no credentials and are what CI
enforces. `terraform plan` against `azurerm` normally needs a subscription, so
CI runs it best-effort (auth failures don't fail the build):

```bash
cd infra/azure/foundry_module
terraform fmt -check -recursive
terraform init -backend=false -input=false
terraform validate
# best-effort, plan-only, never applies:
terraform plan -refresh=false -input=false \
  -var-file=../terraform.tfvars.example || true
```

`scripts/terraform_local_plan.sh` wraps exactly this.

## Applying for real

```bash
cp examples/terraform.tfvars.example terraform.tfvars   # then edit
export ARM_SUBSCRIPTION_ID=... ARM_TENANT_ID=... ARM_CLIENT_ID=... ARM_CLIENT_SECRET=...
terraform init
terraform apply -var-file=terraform.tfvars
# ... run reasoning ...
terraform destroy -var-file=terraform.tfvars
```

The application writes the endpoint key into the vault secret named by
`key_vault_secret_name`; the module intentionally does **not** store secret
values in state.
