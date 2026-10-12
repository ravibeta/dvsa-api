#--------------------------------------
# DVSA solution accelerator
#
# ARM resources are declared here; the project *data plane* (the search indexes
# and the Foundry agents) is registered by core/azure/deploy.py in the same
# apply -- see deploy.tf. The index schema and the agent definitions are not
# duplicated in HCL: they live in core/azure/index_schema.py and
# solution-accelerator/agents/*.yaml so the runtime and the IaC cannot drift.
#--------------------------------------

data "azurerm_client_config" "current" {}

#--------------------------------------
# 1. Resource Group
#--------------------------------------
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group
  location = var.location
}

#--------------------------------------
# 2. Azure AI Search service
#    (the base "dvsa-index" and every per-account "dvsa-index-s-<account>" are
#    created on this service by core/azure -- they are data-plane objects.)
#--------------------------------------
resource "azurerm_search_service" "search" {
  name                = var.search_service_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = var.search_sku
  partition_count     = 1
  replica_count       = 1

  # Semantic ranker: used by the index's "dvsa-semantic" configuration.
  semantic_search_sku = var.search_semantic_sku

  # DVSA authenticates with the admin key and Entra ID.
  local_authentication_enabled = true

  identity {
    type = "SystemAssigned"
  }
}
