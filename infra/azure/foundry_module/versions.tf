terraform {
  required_version = ">= 1.4.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 3.74.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5.0"
    }
  }
}

provider "azurerm" {
  # `features {}` is mandatory. `skip_provider_registration` keeps plan/apply
  # from trying to (re)register resource providers, which also lets `terraform
  # validate` run without a live subscription in CI.
  features {}
  skip_provider_registration = true
}
