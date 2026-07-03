output "foundry_endpoint_url" {
  description = "Scoring/inference endpoint for the reasoning deployment."
  value       = "${azurerm_cognitive_account.foundry.endpoint}openai/deployments/${azurerm_cognitive_deployment.reasoning.name}/chat/completions"
}

output "foundry_deployment_id" {
  description = "Resource id of the reasoning model deployment."
  value       = azurerm_cognitive_deployment.reasoning.id
}

output "foundry_account_name" {
  description = "Name of the Foundry (Azure OpenAI) account."
  value       = azurerm_cognitive_account.foundry.name
}

output "managed_identity_principal_id" {
  description = "Principal id of the user-assigned managed identity."
  value       = azurerm_user_assigned_identity.session.principal_id
}

output "key_vault_name" {
  description = "Key Vault holding the endpoint key."
  value       = azurerm_key_vault.session.name
}

output "key_vault_secret_name" {
  description = "Name of the secret that stores the Foundry endpoint key."
  value       = var.key_vault_secret_name
}

output "resource_group_name" {
  description = "Resource group that is the teardown unit for this session."
  value       = azurerm_resource_group.session.name
}
