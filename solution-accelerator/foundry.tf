#--------------------------------------
# Azure AI Foundry / Azure OpenAI account + model deployments
# Hosts the embedding model (1536-dim vectors) and the chat model used to
# generate captions/labels/tags for each aerial frame.
#--------------------------------------
resource "azurerm_cognitive_account" "foundry" {
  name                  = var.openai_account
  resource_group_name   = azurerm_resource_group.rg.name
  location              = azurerm_resource_group.rg.location
  kind                  = "OpenAI"
  sku_name              = "S0"
  custom_subdomain_name = var.openai_account
}

# Embedding deployment — produces the 1536-dim vectors stored in the index.
resource "azurerm_cognitive_deployment" "embedding" {
  name                 = var.embedding_deployment
  cognitive_account_id = azurerm_cognitive_account.foundry.id

  model {
    format  = "OpenAI"
    name    = var.embedding_model
    version = "2"
  }

  sku {
    name     = "Standard"
    capacity = 1
  }
}

# Chat/caption deployment — generates captions, labels, and tags.
resource "azurerm_cognitive_deployment" "chat" {
  name                 = var.gpt_deployment
  cognitive_account_id = azurerm_cognitive_account.foundry.id

  model {
    format = "OpenAI"
    name   = var.gpt_model
  }

  sku {
    name     = "Standard"
    capacity = 1
  }
}

output "openai_endpoint" {
  value = azurerm_cognitive_account.foundry.endpoint
}

output "openai_primary_key" {
  value     = azurerm_cognitive_account.foundry.primary_access_key
  sensitive = true
}
