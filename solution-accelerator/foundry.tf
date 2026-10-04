#--------------------------------------
# Azure AI Foundry: resource (AIServices) + project + model deployments +
# project connections + RBAC.
#
# Mirrors the live "found-vision-1" resource and its "droneimage" project:
# kind AIServices with project management enabled, the gpt / embedding / mistral
# deployments, and the Azure AI Search connection the search agents reference.
#--------------------------------------
resource "azurerm_cognitive_account" "foundry" {
  name                  = var.openai_account
  resource_group_name   = azurerm_resource_group.rg.name
  location              = azurerm_resource_group.rg.location
  kind                  = "AIServices"
  sku_name              = "S0"
  custom_subdomain_name = var.openai_account

  # Required for Foundry projects to exist under the resource.
  project_management_enabled = true

  identity {
    type = "SystemAssigned"
  }
}

locals {
  # Chat/caption + embedding deployments come from the same variables the
  # runtime config (core/azure/config.py) reads; extras are additive.
  model_deployments = merge(
    {
      (var.gpt_deployment) = {
        model_name    = var.gpt_model
        model_format  = "OpenAI"
        model_version = var.gpt_model_version
        sku_name      = "Standard"
        capacity      = 100
      }
      (var.embedding_deployment) = {
        model_name    = var.embedding_model
        model_format  = "OpenAI"
        model_version = "2"
        sku_name      = "Standard"
        capacity      = 120
      }
    },
    var.extra_model_deployments,
  )

  foundry_endpoint = "https://${var.openai_account}.services.ai.azure.com"
  project_endpoint = "${local.foundry_endpoint}/api/projects/${var.foundry_project}"
}

# Embedding (1536-dim vectors), chat/caption, and the extra models agents use.
resource "azurerm_cognitive_deployment" "models" {
  for_each = local.model_deployments

  name                 = each.key
  cognitive_account_id = azurerm_cognitive_account.foundry.id

  model {
    format  = each.value.model_format
    name    = each.value.model_name
    version = each.value.model_version
  }

  sku {
    name     = each.value.sku_name
    capacity = each.value.capacity
  }
}

#--------------------------------------
# Foundry project (hosts the agents)
#--------------------------------------
resource "azapi_resource" "project" {
  type      = "Microsoft.CognitiveServices/accounts/projects@2025-06-01"
  name      = var.foundry_project
  parent_id = azurerm_cognitive_account.foundry.id
  location  = azurerm_resource_group.rg.location

  identity {
    type = "SystemAssigned"
  }

  body = {
    properties = {
      displayName = var.foundry_project
      description = "DVSA drone video analytics: the connected agents (chat, master, fn, tool, qwen, search) run here."
    }
  }

  depends_on = [azurerm_cognitive_deployment.models]
}

#--------------------------------------
# Project connections
#--------------------------------------

# Azure AI Search -- the master-/scene-search agents' azure_ai_search tool
# resolves its index through this connection (named after the search service).
resource "azapi_resource" "search_connection" {
  type      = "Microsoft.CognitiveServices/accounts/projects/connections@2025-06-01"
  name      = var.search_service_name
  parent_id = azapi_resource.project.id

  body = {
    properties = {
      category      = "CognitiveSearch"
      target        = "https://${azurerm_search_service.search.name}.search.windows.net/"
      authType      = "ApiKey"
      isSharedToAll = false
      credentials = {
        key = azurerm_search_service.search.primary_key
      }
      metadata = {
        ApiType    = "Azure"
        ResourceId = azurerm_search_service.search.id
      }
    }
  }

  # The key is write-only on the service side; don't plan a diff against it.
  lifecycle {
    ignore_changes = [body]
  }
}

# Optional model-gateway connection registering the managed-compute Qwen
# deployment with the project (enabled by supplying qwen_gateway_api_key). The
# deployment itself (qwen_model_deployment) is a managed-compute data-plane
# object created in the Foundry portal/CLI -- core/azure/deploy.py verifies it.
resource "azapi_resource" "qwen_gateway_connection" {
  count = var.qwen_gateway_api_key == "" ? 0 : 1

  type      = "Microsoft.CognitiveServices/accounts/projects/connections@2025-06-01"
  name      = "qwen-agent-in-a-team"
  parent_id = azapi_resource.project.id

  body = {
    properties = {
      category      = "ModelGateway"
      target        = "${azurerm_cognitive_account.foundry.endpoint}openai/v1"
      authType      = "ApiKey"
      isSharedToAll = false
      credentials = {
        key = var.qwen_gateway_api_key
      }
      metadata = {
        deploymentInPath = "false"
        customHeaders    = "{}"
        models = jsonencode([{
          name       = var.qwen_model_deployment
          properties = { model = { name = var.qwen_model_name, format = "Custom" } }
        }])
      }
    }
  }
}

#--------------------------------------
# RBAC: who may create/run agents in the project (data-plane "Foundry User").
# The deployer is always included so the data-plane registration in
# deploy.tf is authorised.
#--------------------------------------
resource "azurerm_role_assignment" "foundry_user" {
  for_each = toset(concat([data.azurerm_client_config.current.object_id], var.foundry_user_principal_ids))

  scope              = azapi_resource.project.id
  role_definition_id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/providers/Microsoft.Authorization/roleDefinitions/${var.foundry_user_role_id}"
  principal_id       = each.value
}
