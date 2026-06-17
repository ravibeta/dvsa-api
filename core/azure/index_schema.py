"""Azure AI Search index schema for aerial frames.

The schema stores one document per aerial frame with a 1536-dim embedding plus
the descriptive fields required by the spec: caption, labels, tags, the user
identifier, and the blob path of the frame in the ``sadronevideo`` account.

Two representations are provided:

- :func:`field_spec` / :func:`describe_index` — plain ``dict`` descriptions that
  need **no** SDK. Used by the dry-run provisioner, tests, and docs.
- :func:`build_search_index` — a concrete ``azure.search.documents.indexes``
  ``SearchIndex`` (HNSW vector search), imported lazily so this module loads
  without ``azure-search-documents`` installed.
"""

from __future__ import annotations

from typing import Any, Dict, List

VECTOR_PROFILE = "dvsa-hnsw-profile"
VECTOR_ALGORITHM = "dvsa-hnsw"


def field_spec(dimensions: int = 1536) -> List[Dict[str, Any]]:
    """SDK-free description of the index fields (order is significant)."""
    return [
        {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
        # 1536-dim embedding per aerial frame.
        {
            "name": "vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "dimensions": dimensions,
            "vector_search_profile": VECTOR_PROFILE,
        },
        {"name": "caption", "type": "Edm.String", "searchable": True},
        {"name": "labels", "type": "Collection(Edm.String)", "filterable": True,
         "facetable": True, "searchable": True},
        {"name": "tags", "type": "Collection(Edm.String)", "filterable": True,
         "facetable": True, "searchable": True},
        # User identifier — also the per-session filter key for "filter" mode.
        {"name": "user", "type": "Edm.String", "filterable": True,
         "facetable": True},
        {"name": "session", "type": "Edm.String", "filterable": True,
         "facetable": True},
        # path/to/file in the sadronevideo storage account (blob path).
        {"name": "path", "type": "Edm.String", "filterable": True},
        {"name": "created", "type": "Edm.DateTimeOffset", "filterable": True,
         "sortable": True},
    ]


def describe_index(name: str, dimensions: int = 1536) -> Dict[str, Any]:
    """Full SDK-free index description (used by dry-run + assertions)."""
    return {
        "name": name,
        "fields": field_spec(dimensions),
        "vector_search": {
            "algorithm": VECTOR_ALGORITHM,
            "kind": "hnsw",
            "metric": "cosine",
            "profile": VECTOR_PROFILE,
        },
    }


def build_search_index(name: str, dimensions: int = 1536):
    """Build a concrete ``SearchIndex`` (lazy import of azure-search-documents)."""
    from azure.search.documents.indexes.models import (  # noqa: PLC0415
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True,
                    filterable=True),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
        SearchableField(name="caption", type=SearchFieldDataType.String),
        SearchableField(
            name="labels",
            collection=True,
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True, facetable=True,
        ),
        SearchableField(
            name="tags",
            collection=True,
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True, facetable=True,
        ),
        SimpleField(name="user", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="session", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="path", type=SearchFieldDataType.String,
                    filterable=True),
        SimpleField(name="created", type=SearchFieldDataType.DateTimeOffset,
                    filterable=True, sortable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=VECTOR_ALGORITHM,
                parameters=HnswParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE,
                    m=4, ef_construction=400, ef_search=500,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE, algorithm_configuration_name=VECTOR_ALGORITHM
            )
        ],
    )

    return SearchIndex(name=name, fields=fields, vector_search=vector_search)
