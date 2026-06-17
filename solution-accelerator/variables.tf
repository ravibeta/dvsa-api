# Input variables for the DVSA session-scoped Azure environment.
# These are supplied by core/azure/terraform.py (-var ...) or via *.tfvars.

variable "resource_group" {
  type        = string
  description = "Resource group that holds all DVSA resources."
  default     = "rg-dvsa"
}

variable "location" {
  type        = string
  description = "Azure region for all resources."
  default     = "East US"
}

# --- Storage ---------------------------------------------------------------
variable "storage_account" {
  type        = string
  description = "Globally-unique storage account name for aerial video."
  default     = "sadronevideo"
}

variable "input_container" {
  type        = string
  description = "Container holding ingested aerial frames."
  default     = "input"
}

variable "output_container" {
  type        = string
  description = "Container holding derived artifacts."
  default     = "output"
}

# --- AI Search -------------------------------------------------------------
variable "search_service_name" {
  type        = string
  description = "Azure AI Search service name."
  default     = "srch-dvsa"
}

variable "search_sku" {
  type        = string
  description = "AI Search SKU."
  default     = "standard"
}

variable "search_index_name" {
  type        = string
  description = "Base AI Search index name."
  default     = "dvsa-index"
}

variable "vector_dimensions" {
  type        = number
  description = "Embedding width stored per aerial frame."
  default     = 1536
}

# --- Foundry / Azure OpenAI ------------------------------------------------
variable "openai_account" {
  type        = string
  description = "Azure OpenAI (AI Foundry) account name."
  default     = "dvsa-foundry"
}

variable "gpt_deployment" {
  type        = string
  description = "Chat/caption model deployment name."
  default     = "gpt-4o-mini"
}

variable "gpt_model" {
  type        = string
  description = "Chat/caption model name."
  default     = "gpt-4o-mini"
}

variable "embedding_deployment" {
  type        = string
  description = "Embedding model deployment name."
  default     = "text-embedding-ada-002"
}

variable "embedding_model" {
  type        = string
  description = "Embedding model name."
  default     = "text-embedding-ada-002"
}
