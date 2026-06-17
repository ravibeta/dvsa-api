#--------------------------------------
# Storage account "sadronevideo" + input/output containers
# Stores aerial video frames (input) and derived artifacts (output).
#--------------------------------------
resource "azurerm_storage_account" "sadronevideo" {
  name                     = var.storage_account
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  min_tls_version          = "TLS1_2"
}

resource "azurerm_storage_container" "input" {
  name                  = var.input_container
  storage_account_name  = azurerm_storage_account.sadronevideo.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "output" {
  name                  = var.output_container
  storage_account_name  = azurerm_storage_account.sadronevideo.name
  container_access_type = "private"
}

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
