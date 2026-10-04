terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 4.30.0, < 6.0.0"
    }
    # Foundry projects and project connections are ARM child resources that
    # azurerm does not model; azapi manages them.
    azapi = {
      source  = "Azure/azapi"
      version = "~> 2.4"
    }
  }
}

provider "azurerm" {
  features {}
  # null -> falls back to ARM_SUBSCRIPTION_ID / the az CLI context.
  subscription_id = var.subscription_id
}

provider "azapi" {
  subscription_id = var.subscription_id
}
