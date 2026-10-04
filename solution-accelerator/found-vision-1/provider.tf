provider "azurerm" {
  features {
  }
  subscription_id                 = "656e67c6-f810-4ea6-8b89-636dd0b6774c"
  environment                     = "public"
  use_msi                         = false
  use_cli                         = true
  use_oidc                        = false
  resource_provider_registrations = "none"
}
