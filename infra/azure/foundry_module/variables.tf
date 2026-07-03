# Input variables for one ephemeral Azure Foundry reasoning session.
# Everything is scoped to a single resource group — the module never creates
# subscription-level resources.

variable "subscription_id" {
  type        = string
  description = "Target subscription id (informational; auth via ARM_* / CLI)."
  default     = ""
}

variable "location" {
  type        = string
  description = "Azure region for all session resources."
  default     = "eastus"
}

variable "resource_group_name" {
  type        = string
  description = "Resource group that holds the ephemeral session resources."
}

variable "session_name" {
  type        = string
  description = "Short, unique, DNS-safe session slug used to name resources."

  validation {
    condition     = can(regex("^[a-z0-9-]{1,24}$", var.session_name))
    error_message = "session_name must be 1-24 chars of lowercase letters, digits or dashes."
  }
}

variable "foundry_model_sku" {
  type        = string
  description = "Foundry / Azure OpenAI reasoning model to deploy (e.g. o1-mini)."
  default     = "o1-mini"
}

variable "foundry_model_version" {
  type        = string
  description = "Model version for the reasoning deployment."
  default     = "1"
}

variable "managed_identity_name" {
  type        = string
  description = "Name of the user-assigned managed identity for the deployment."
}

variable "key_vault_name" {
  type        = string
  description = "Key Vault name (globally unique). Empty -> derived from session_name."
  default     = ""
}

variable "key_vault_secret_name" {
  type        = string
  description = "Name of the secret that will hold the Foundry endpoint key."
  default     = "foundry-endpoint-key"
}

variable "private_networking" {
  type        = bool
  description = "When true, provision a VNet/subnet and keep data-plane private."
  default     = false
}

variable "vnet_id" {
  type        = string
  description = "Optional existing VNet id to attach to (unused when empty)."
  default     = ""
}

variable "ttl_minutes" {
  type        = number
  description = "Session TTL in minutes (recorded as a tag for cost auditing)."
  default     = 30

  validation {
    condition     = var.ttl_minutes > 0 && var.ttl_minutes <= 1440
    error_message = "ttl_minutes must be between 1 and 1440."
  }
}

variable "cost_center" {
  type        = string
  description = "Cost-center tag value for billing/showback."
  default     = "dvsa-reasoning"
}

variable "tags" {
  type        = map(string)
  description = "Additional tags merged onto every resource."
  default     = {}
}
