#--------------------------------------
# One-shot data-plane registration.
#
# Search indexes and Foundry agents are project data-plane objects, not ARM
# resources, so they are registered by core/azure/deploy.py in the same
# `terraform apply`, after everything it depends on exists. A single run:
#   1. creates/updates the base index (var.search_index_name, "dvsa-index") from
#      core/azure/index_schema.py -- the same builder the runtime uses for each
#      per-account "dvsa-index-s-<account>" -- with HNSW + exhaustive-KNN
#      vector profiles, BM25, semantic ranking and the context/ACL fields;
#   2. registers every agent in solution-accelerator/agents/*.yaml (function
#      tools generated from core/azure/analyzer.py) in both the versioned and
#      classic agent stores, so all agents exist before the first /chat query;
#   3. verifies the model deployments those agents use.
#
# It re-runs whenever the index schema, agent definitions, or tool code change.
#--------------------------------------
locals {
  repo_root = abspath("${path.module}/..")

  dataplane_inputs = concat(
    [for f in sort(tolist(fileset("${path.module}/agents", "*.yaml"))) : filesha256("${path.module}/agents/${f}")],
    [
      filesha256("${local.repo_root}/core/azure/index_schema.py"),
      filesha256("${local.repo_root}/core/azure/analyzer.py"),
      filesha256("${local.repo_root}/core/azure/foundry_provisioning.py"),
      filesha256("${local.repo_root}/core/azure/deploy.py"),
    ],
  )
}

resource "terraform_data" "dataplane" {
  count = var.run_dataplane_deploy ? 1 : 0

  triggers_replace = {
    inputs  = sha256(join(",", local.dataplane_inputs))
    project = azapi_resource.project.id
    index   = var.search_index_name
    search  = azurerm_search_service.search.id
  }

  provisioner "local-exec" {
    working_dir = local.repo_root
    # --retries rides out role-assignment propagation (can take minutes).
    command = "${var.python_executable} -m core.azure.deploy --retries 12"

    environment = {
      DJANGO_SETTINGS_MODULE            = "config.settings.development"
      AZURE_SUBSCRIPTION_ID             = data.azurerm_client_config.current.subscription_id
      AZURE_RESOURCE_GROUP              = azurerm_resource_group.rg.name
      AZURE_PROJECT_ENDPOINT            = local.project_endpoint
      AZURE_SEARCH_ENDPOINT             = "https://${azurerm_search_service.search.name}.search.windows.net"
      AZURE_SEARCH_ADMIN_KEY            = azurerm_search_service.search.primary_key
      AZURE_SEARCH_INDEX_NAME           = var.search_index_name
      AZURE_SEARCH_SERVICE_NAME         = azurerm_search_service.search.name
      AZURE_OPENAI_ENDPOINT             = azurerm_cognitive_account.foundry.endpoint
      AZURE_OPENAI_API_KEY              = azurerm_cognitive_account.foundry.primary_access_key
      AZURE_OPENAI_GPT_DEPLOYMENT       = var.gpt_deployment
      AZURE_OPENAI_EMBEDDING_DEPLOYMENT = var.embedding_deployment
      AZURE_OPENAI_EMBEDDING_MODEL      = var.embedding_model
    }
  }

  depends_on = [
    azurerm_cognitive_deployment.models,
    azapi_resource.search_connection,
    azurerm_role_assignment.foundry_user,
  ]
}
