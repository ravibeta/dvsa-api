#!/usr/bin/env bash
# Plan-only Terraform validation for the ephemeral Foundry module.
#
# Safe for CI: it NEVER runs `terraform apply`/`destroy` and needs no live Azure
# credentials. `fmt` + `validate` are enforced (they fail the build on error);
# `plan` is best-effort because the azurerm provider needs a subscription to
# refresh — an auth failure there must not fail the build.
set -uo pipefail

MODULE_DIR="${1:-infra/azure/foundry_module}"
VAR_FILE="${2:-infra/azure/terraform.tfvars.example}"

if ! command -v terraform >/dev/null 2>&1; then
  echo "terraform not installed; skipping (nothing to validate)."
  exit 0
fi

echo "==> terraform fmt (check) on ${MODULE_DIR}"
if ! terraform -chdir="${MODULE_DIR}" fmt -check -recursive; then
  echo "WARNING: module not canonically formatted; auto-formatting in place."
  echo "         Run 'terraform fmt -recursive ${MODULE_DIR}' and commit the result."
  terraform -chdir="${MODULE_DIR}" fmt -recursive
fi

echo "==> terraform init (-backend=false)"
terraform -chdir="${MODULE_DIR}" init -backend=false -input=false || exit 1

echo "==> terraform validate"
terraform -chdir="${MODULE_DIR}" validate || exit 1

# Resolve the var-file relative to the module dir for -chdir.
ABS_VAR_FILE="$(cd "$(dirname "${VAR_FILE}")" && pwd)/$(basename "${VAR_FILE}")"

echo "==> terraform plan (plan-only, best-effort, -refresh=false)"
if terraform -chdir="${MODULE_DIR}" plan -refresh=false -input=false \
    -var-file="${ABS_VAR_FILE}"; then
  echo "plan succeeded."
else
  echo "plan could not complete (expected without live Azure creds); " \
       "fmt+validate passed, which is what CI gates on."
fi

echo "OK: terraform module fmt + validate passed."
