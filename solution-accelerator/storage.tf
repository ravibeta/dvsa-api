#--------------------------------------
# Storage account "sadronevideo" + input/output containers
# Stores aerial video + extracted frames (input) and derived artifacts (output).
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
  storage_account_id    = azurerm_storage_account.sadronevideo.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "output" {
  name                  = var.output_container
  storage_account_id    = azurerm_storage_account.sadronevideo.id
  container_access_type = "private"
}
