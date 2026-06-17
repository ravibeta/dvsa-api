terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">=3.74.0"
    }
  }
}

provider "azurerm" {
  features {}
}

#--------------------------------------
# 1. Resource Group
#--------------------------------------
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group
  location = var.location
}

#--------------------------------------
# 2. Azure Cognitive Search Service
#--------------------------------------
resource "azurerm_search_service" "search" {
  name                = var.search_service_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = var.search_sku
  partition_count     = 1
  replica_count       = 1
}

#--------------------------------------
# 3. AI Search Index Definition
#--------------------------------------
resource "azurerm_search_index" "index007" {
  name                = var.search_index_name
  service_name        = azurerm_search_service.search.name
  resource_group_name = azurerm_resource_group.rg.name

  field {
    name         = "id"
    type         = "Edm.String"
    key          = true
    retrievable  = true
    searchable   = false
  }

  field {
    name        = "description"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
  }

  field {
    name                  = "vector"
    type                  = "Collection(Edm.Single)"
    searchable            = true
    retrievable           = true
    dimensions            = var.vector_dimensions
    vector_search_profile = "myHnswProfile"
  }

  field {
    name        = "objects"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
    analyzer    = "standard.lucene"
  }

  field {
    name        = "tags"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
    analyzer    = "standard.lucene"
  }

  field {
    name        = "title"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
    analyzer    = "standard.lucene"
  }

  field {
    name        = "imageurl"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
    analyzer    = "standard.lucene"
  }

  field {
    name        = "location"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
    analyzer    = "standard.lucene"
  }

  field {
    name        = "boundingbox"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
    analyzer    = "standard.lucene"
  }

  field {
    name        = "created"
    type        = "Edm.DateTimeOffset"
    searchable  = false
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
  }

  field {
    name        = "account_id"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
    analyzer    = "standard.lucene"
  }

  field {
    name        = "geotags"
    type        = "Edm.String"
    searchable  = true
    filterable  = true
    retrievable = true
    sortable    = true
    facetable   = true
    analyzer    = "standard.lucene"
  }

  #--------------------------------------
  # Vector Search Configuration
  #--------------------------------------
  vector_search {
    algorithm {
      name = "myHnsw"
      kind = "hnsw"

      hnsw_parameters {
        metric          = "cosine"
        m               = 4
        ef_construction = 400
        ef_search       = 1000
      }
    }

    algorithm {
      name = "myExhaustiveKnn"
      kind = "exhaustiveKnn"

      exhaustive_knn_parameters {
        metric = "cosine"
      }
    }

    profile {
      name      = "myHnswProfile"
      algorithm = "myHnsw"
      vectorizer = "vectorizer-1748574121416"
    }

    profile {
      name      = "myExhaustiveKnnProfile"
      algorithm = "myExhaustiveKnn"
      vectorizer = "vectorizer-1748574121416"
    }

    vectorizer {
      name = "vectorizer-1748574121416"
      kind = "azureOpenAI"

      azure_openai_parameters {
        resource_uri  = "https://openvision.openai.azure.com"
        deployment_id = "text-embedding-ada-002"
      }
    }
  }

  #--------------------------------------
  # Similarity & Semantic Configurations
  #--------------------------------------
  similarity {
    odatatype = "#Microsoft.Azure.Search.BM25Similarity"
  }

  semantic_settings {
    default_configuration = "mysemantic"

    configuration {
      name = "mysemantic"

      prioritized_fields {
        title_field {
          field_name = "description"
        }

        prioritized_content_fields {
          field_name = "id"
        }

        prioritized_content_fields {
          field_name = "description"
        }

        prioritized_keywords_fields {
          field_name = "id"
        }

        prioritized_keywords_fields {
          field_name = "description"
        }
      }
    }
  }
}
