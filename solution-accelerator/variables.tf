# Input variables for the DVSA session-scoped Azure environment.
# These are supplied by core/azure/terraform.py (-var ...) or via *.tfvars.

variable "subscription_id" {
  type        = string
  description = "Subscription to deploy into (null: ARM_SUBSCRIPTION_ID / az CLI context)."
  default     = null
}

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

variable "gpt_model_version" {
  type        = string
  description = "Chat/caption model version."
  default     = "2025-04-14"
}

# --- AI Search extras ------------------------------------------------------
variable "search_semantic_sku" {
  type        = string
  description = "Semantic ranker tier for the search service (free | standard | disabled)."
  default     = "free"
}

# --- Foundry project -------------------------------------------------------
variable "foundry_project" {
  type        = string
  description = "Foundry project that hosts the agents."
  default     = "droneimage"
}

variable "extra_model_deployments" {
  description = "Additional model deployments on the Foundry resource (the gpt + embedding deployments above are always created)."
  type = map(object({
    model_name    = string
    model_format  = string
    model_version = string
    sku_name      = string
    capacity      = number
  }))
  default = {
    "text-embedding-3-large" = {
      model_name    = "text-embedding-3-large"
      model_format  = "OpenAI"
      model_version = "1"
      sku_name      = "GlobalStandard"
      capacity      = 500
    }
    "mistral-medium-2505" = {
      model_name    = "mistral-medium-2505"
      model_format  = "Mistral AI"
      model_version = "1"
      sku_name      = "GlobalStandard"
      capacity      = 20
    }
  }
}

variable "foundry_user_principal_ids" {
  type        = list(string)
  description = "Extra principals (object ids) granted data-plane access to the project; the deployer always is."
  default     = []
}

variable "foundry_user_role_id" {
  type        = string
  description = "Role definition id for project data-plane access (built-in 'Foundry User', formerly 'Azure AI User')."
  default     = "53ca6127-db72-4b80-b1b0-d745d6d5456d"
}

# --- Qwen vision agent -----------------------------------------------------
variable "qwen_model_deployment" {
  type        = string
  description = "Managed-compute Qwen deployment name in the project (verified, not created, by deploy)."
  default     = "qwen--qwen35-08b-2"
}

variable "qwen_model_name" {
  type        = string
  description = "Model behind the Qwen deployment."
  default     = "qwen--qwen3.5-0.8b"
}

variable "qwen_gateway_api_key" {
  type        = string
  description = "API key for the Qwen model-gateway project connection; empty skips creating the connection."
  default     = ""
  sensitive   = true
}

# --- One-shot data-plane deploy --------------------------------------------
variable "run_dataplane_deploy" {
  type        = bool
  description = "Register the search index + Foundry agents (core.azure.deploy) as part of apply."
  default     = true
}

variable "python_executable" {
  type        = string
  description = "Python interpreter with the dvsa-api requirements installed."
  default     = "python"
}
