output "storage_account_name" {
  value = azurerm_storage_account.sadronevideo.name
}

output "storage_primary_blob_endpoint" {
  value = azurerm_storage_account.sadronevideo.primary_blob_endpoint
}

output "storage_primary_connection_string" {
  value     = azurerm_storage_account.sadronevideo.primary_connection_string
  sensitive = true
}

output "search_endpoint" {
  value = "https://${azurerm_search_service.search.name}.search.windows.net"
}

output "search_index_name" {
  description = "Base index; per-account indexes are <name>-s-<account>."
  value       = var.search_index_name
}

output "openai_endpoint" {
  value = azurerm_cognitive_account.foundry.endpoint
}

output "openai_primary_key" {
  value     = azurerm_cognitive_account.foundry.primary_access_key
  sensitive = true
}

output "foundry_project_endpoint" {
  description = "AZURE_PROJECT_ENDPOINT for dvsa-api."
  value       = local.project_endpoint
}
