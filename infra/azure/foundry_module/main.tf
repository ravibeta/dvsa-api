#############################################################################
# Ephemeral Azure Foundry reasoning session
#
# Creates, inside ONE resource group, everything a per-session reasoning
# deployment needs: a Foundry (Azure OpenAI) account + reasoning deployment, a
# user-assigned managed identity, a Key Vault for the endpoint key with
# least-privilege RBAC, and (optionally) a private VNet/subnet. Torn down as a
# unit via `terraform destroy`.
#############################################################################

data "azurerm_client_config" "current" {}

locals {
  # Merge standard cost/tracking tags onto every resource.
  base_tags = merge(
    {
      component   = "dvsa-foundry-reasoning"
      session     = var.session_name
      cost_center = var.cost_center
      ttl_minutes = tostring(var.ttl_minutes)
      managed_by  = "terraform"
    },
    var.tags,
  )

  # Key Vault names are globally unique and <= 24 chars; derive if not given.
  # coalesce() skips the empty default, falling back to the derived name.
  key_vault_name = coalesce(
    var.key_vault_name,
    substr("kv${replace(var.session_name, "-", "")}", 0, 24),
  )

  foundry_account_name = substr("foundry-${var.session_name}", 0, 60)
}

# --- Resource group (the teardown unit) ------------------------------------
resource "azurerm_resource_group" "session" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.base_tags
}

# --- User-assigned managed identity for the deployment ---------------------
resource "azurerm_user_assigned_identity" "session" {
  name                = var.managed_identity_name
  resource_group_name = azurerm_resource_group.session.name
  location            = azurerm_resource_group.session.location
  tags                = local.base_tags
}

# --- Optional private networking -------------------------------------------
resource "azurerm_virtual_network" "session" {
  count               = var.private_networking ? 1 : 0
  name                = "vnet-${var.session_name}"
  resource_group_name = azurerm_resource_group.session.name
  location            = azurerm_resource_group.session.location
  address_space       = ["10.60.0.0/24"]
  tags                = local.base_tags
}

resource "azurerm_subnet" "session" {
  count                = var.private_networking ? 1 : 0
  name                 = "snet-${var.session_name}"
  resource_group_name  = azurerm_resource_group.session.name
  virtual_network_name = azurerm_virtual_network.session[0].name
  address_prefixes     = ["10.60.0.0/26"]

  service_endpoints = ["Microsoft.KeyVault", "Microsoft.CognitiveServices"]
}

# --- Key Vault (RBAC-authorized) for the endpoint key ----------------------
resource "azurerm_key_vault" "session" {
  name                       = local.key_vault_name
  resource_group_name        = azurerm_resource_group.session.name
  location                   = azurerm_resource_group.session.location
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  enable_rbac_authorization  = true
  purge_protection_enabled   = false
  soft_delete_retention_days = 7
  tags                       = local.base_tags

  # Lock down the data plane when private networking is requested.
  dynamic "network_acls" {
    for_each = var.private_networking ? [1] : []
    content {
      bypass                     = "AzureServices"
      default_action             = "Deny"
      virtual_network_subnet_ids = [azurerm_subnet.session[0].id]
    }
  }
}

# Least privilege: the managed identity may only READ secrets in this vault.
resource "azurerm_role_assignment" "identity_kv_secrets_user" {
  scope                = azurerm_key_vault.session.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.session.principal_id
}

# The provisioning principal needs to WRITE the endpoint key once.
resource "azurerm_role_assignment" "deployer_kv_secrets_officer" {
  scope                = azurerm_key_vault.session.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

# --- Foundry (Azure OpenAI) account + reasoning deployment -----------------
resource "azurerm_cognitive_account" "foundry" {
  name                  = local.foundry_account_name
  resource_group_name   = azurerm_resource_group.session.name
  location              = azurerm_resource_group.session.location
  kind                  = "OpenAI"
  sku_name              = "S0"
  custom_subdomain_name = local.foundry_account_name

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.session.id]
  }

  tags = local.base_tags
}

resource "azurerm_cognitive_deployment" "reasoning" {
  name                 = var.foundry_model_sku
  cognitive_account_id = azurerm_cognitive_account.foundry.id

  model {
    format  = "OpenAI"
    name    = var.foundry_model_sku
    version = var.foundry_model_version
  }

  sku {
    name     = "Standard"
    capacity = 1
  }
}
