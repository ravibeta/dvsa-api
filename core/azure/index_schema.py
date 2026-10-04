"""Azure AI Search index schema for aerial frames.

The schema stores one document per aerial frame with a 1536-dim embedding plus
the descriptive fields required by the spec: caption, labels, tags, annotations,
location, timestamp, the user identifier, and the blob path of the frame in the
``sadronevideo`` account. It is the single source of truth for *every* index
DVSA creates: the shared ``dvsa-index`` provisioned at deploy time and each
per-account ``dvsa-index-s-<account>`` created at runtime.

Retrieval capabilities (see ``solution-accelerator/Vector-Search-Index-Retrieval-Settings``)
are declared here, on the index, so any query path benefits without query-time
changes:

- **Dense retrieval**: one 1536-dim ``vector`` field with *both* an HNSW profile
  (fast approximate search, the field's default) and an exhaustive-KNN profile
  (exact search — select it per query with ``exhaustive=True`` on the vector
  query; a field can bind only one profile, so exact search reuses the same
  field). An Azure OpenAI vectorizer lets queries embed text server-side.
- **Lexical retrieval**: BM25 similarity over ``searchable`` text fields, with a
  scoring profile that boosts captions/tags/labels/annotations. Dense + lexical
  results are fused by Azure AI Search (RRF) when a query sends both.
- **Reranking**: a semantic configuration (title/content/keyword fields) for the
  semantic ranker.
- **Context assembly**: parent-document metadata (``video_id``, ``title``,
  ``section``), adjacent chunks (``frame_number``, ``prev_id``, ``next_id``),
  and ``version`` + ACL (``acl_users``/``acl_groups``) fields so callers can
  version- and permission-filter before building context.

Two representations are provided:

- :func:`field_spec` / :func:`describe_index` — plain ``dict`` descriptions that
  need **no** SDK. Used by the dry-run provisioner, tests, and docs.
- :func:`build_search_index` — a concrete ``azure.search.documents.indexes``
  ``SearchIndex``, imported lazily so this module loads without
  ``azure-search-documents`` installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

VECTOR_PROFILE = "dvsa-hnsw-profile"          # default (approximate) profile
VECTOR_PROFILE_EXHAUSTIVE = "dvsa-eknn-profile"
VECTOR_ALGORITHM = "dvsa-hnsw"
VECTOR_ALGORITHM_EXHAUSTIVE = "dvsa-eknn"
VECTORIZER_NAME = "dvsa-aoai-vectorizer"
SEMANTIC_CONFIG = "dvsa-semantic"
SCORING_PROFILE = "dvsa-boost"

# Searchable text fields and the BM25 weight the scoring profile gives each.
TEXT_WEIGHTS: Dict[str, float] = {
    "caption": 3.0,
    "tags": 2.0,
    "labels": 2.0,
    "annotations": 1.5,
    "description": 1.0,
    "objects": 1.0,
    "location": 0.5,
}


def aoai_resource_uri(endpoint: Optional[str]) -> Optional[str]:
    """Bare Azure OpenAI resource host for a search vectorizer ``resourceUri``.

    Azure AI Search builds its own ``{resourceUri}/openai/deployments/...``
    path, so the unified ``/openai/v1`` suffix some configs carry must go.
    """
    if not endpoint:
        return None
    uri = endpoint.rstrip("/")
    if uri.lower().endswith("/openai/v1"):
        uri = uri[: -len("/openai/v1")]
    return uri


def field_spec(dimensions: int = 1536) -> List[Dict[str, Any]]:
    """SDK-free description of the index fields (order is significant)."""
    s = "Edm.String"
    coll = "Collection(Edm.String)"
    return [
        {"name": "id", "type": s, "key": True, "filterable": True},
        # 1536-dim embedding per aerial frame.
        {
            "name": "vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "dimensions": dimensions,
            "vector_search_profile": VECTOR_PROFILE,
        },
        {"name": "caption", "type": s, "searchable": True},
        {"name": "labels", "type": coll, "filterable": True,
         "facetable": True, "searchable": True},
        {"name": "tags", "type": coll, "filterable": True,
         "facetable": True, "searchable": True},
        {"name": "annotations", "type": s, "searchable": True},
        # User identifier — also the per-session filter key for "filter" mode.
        {"name": "user", "type": s, "filterable": True, "facetable": True},
        {"name": "session", "type": s, "filterable": True, "facetable": True},
        # path/to/file in the sadronevideo storage account (blob path).
        {"name": "path", "type": s, "filterable": True},
        {"name": "created", "type": "Edm.DateTimeOffset", "filterable": True,
         "sortable": True},
        # When in the source video the frame occurs / was captured.
        {"name": "timestamp", "type": "Edm.DateTimeOffset", "filterable": True,
         "sortable": True},
        # --- ported-from-ezvision fidelity fields ------------------------
        # account_id mirrors the source index007 partition key; the agentic
        # retrieval filter is `account_id eq '<id>'`.
        {"name": "account_id", "type": s, "filterable": True, "facetable": True},
        {"name": "description", "type": s, "searchable": True},
        {"name": "objects", "type": s, "searchable": True, "filterable": True},
        {"name": "boundingbox", "type": s, "filterable": True},
        {"name": "geotags", "type": s, "filterable": True},
        {"name": "location", "type": s, "searchable": True, "filterable": True,
         "facetable": True},
        {"name": "imageurl", "type": s, "filterable": True},
        # --- context assembly: parent metadata, section, adjacent chunks --
        {"name": "video_id", "type": s, "filterable": True, "facetable": True},
        {"name": "title", "type": s, "searchable": True, "filterable": True},
        {"name": "section", "type": s, "searchable": True, "filterable": True},
        {"name": "frame_number", "type": "Edm.Int32", "filterable": True,
         "sortable": True},
        {"name": "prev_id", "type": s, "filterable": True},
        {"name": "next_id", "type": s, "filterable": True},
        # --- version + ACL checks ----------------------------------------
        {"name": "version", "type": s, "filterable": True},
        {"name": "acl_users", "type": coll, "filterable": True},
        {"name": "acl_groups", "type": coll, "filterable": True},
    ]


def vector_search_spec(
    *, openai_endpoint: Optional[str] = None,
    embedding_deployment: str = "text-embedding-ada-002",
    embedding_model: str = "text-embedding-ada-002",
) -> Dict[str, Any]:
    """SDK-free description of the vector search configuration."""
    spec: Dict[str, Any] = {
        "algorithms": [
            {"name": VECTOR_ALGORITHM, "kind": "hnsw", "metric": "cosine",
             "m": 4, "ef_construction": 400, "ef_search": 500},
            {"name": VECTOR_ALGORITHM_EXHAUSTIVE, "kind": "exhaustiveKnn",
             "metric": "cosine"},
        ],
        "profiles": [
            {"name": VECTOR_PROFILE, "algorithm": VECTOR_ALGORITHM},
            {"name": VECTOR_PROFILE_EXHAUSTIVE,
             "algorithm": VECTOR_ALGORITHM_EXHAUSTIVE},
        ],
        # Back-compat keys (single default algorithm/profile).
        "algorithm": VECTOR_ALGORITHM, "kind": "hnsw", "metric": "cosine",
        "profile": VECTOR_PROFILE,
    }
    uri = aoai_resource_uri(openai_endpoint)
    if uri:
        for p in spec["profiles"]:
            p["vectorizer"] = VECTORIZER_NAME
        spec["vectorizers"] = [{
            "name": VECTORIZER_NAME, "kind": "azureOpenAI",
            "resource_uri": uri, "deployment_id": embedding_deployment,
            "model_name": embedding_model,
        }]
    return spec


def describe_index(name: str, dimensions: int = 1536, **vector_kwargs: Any) -> Dict[str, Any]:
    """Full SDK-free index description (used by dry-run + assertions)."""
    return {
        "name": name,
        "fields": field_spec(dimensions),
        "vector_search": vector_search_spec(**vector_kwargs),
        "similarity": {"kind": "bm25"},
        "semantic": {
            "default_configuration": SEMANTIC_CONFIG,
            "title_field": "title",
            "content_fields": ["caption", "description", "annotations"],
            "keywords_fields": ["tags", "labels"],
        },
        "scoring_profile": {"name": SCORING_PROFILE, "text_weights": TEXT_WEIGHTS},
    }


def build_search_index(
    name: str, dimensions: int = 1536, *,
    openai_endpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    embedding_deployment: str = "text-embedding-ada-002",
    embedding_model: str = "text-embedding-ada-002",
):
    """Build a concrete ``SearchIndex`` (lazy import of azure-search-documents).

    When ``openai_endpoint`` is given, an Azure OpenAI vectorizer is attached to
    both vector profiles so text queries can be embedded server-side; with no
    ``openai_api_key`` the search service's managed identity is used.
    """
    from azure.search.documents.indexes.models import (  # noqa: PLC0415
        AzureOpenAIVectorizer,
        AzureOpenAIVectorizerParameters,
        BM25SimilarityAlgorithm,
        ExhaustiveKnnAlgorithmConfiguration,
        ExhaustiveKnnParameters,
        HnswAlgorithmConfiguration,
        HnswParameters,
        ScoringProfile,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchableField,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        TextWeights,
        VectorSearch,
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )

    String = SearchFieldDataType.String
    StringColl = SearchFieldDataType.Collection(SearchFieldDataType.String)

    def text(n: str, **kw: Any) -> Any:
        return SearchableField(name=n, type=String, **kw)

    def coll(n: str, **kw: Any) -> Any:
        return SearchableField(name=n, collection=True, type=StringColl, **kw)

    fields = [
        SimpleField(name="id", type=String, key=True, filterable=True),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
        text("caption"),
        coll("labels", filterable=True, facetable=True),
        coll("tags", filterable=True, facetable=True),
        text("annotations"),
        SimpleField(name="user", type=String, filterable=True, facetable=True),
        SimpleField(name="session", type=String, filterable=True, facetable=True),
        SimpleField(name="path", type=String, filterable=True),
        SimpleField(name="created", type=SearchFieldDataType.DateTimeOffset,
                    filterable=True, sortable=True),
        SimpleField(name="timestamp", type=SearchFieldDataType.DateTimeOffset,
                    filterable=True, sortable=True),
        SimpleField(name="account_id", type=String, filterable=True, facetable=True),
        text("description"),
        text("objects", filterable=True),
        SimpleField(name="boundingbox", type=String, filterable=True),
        SimpleField(name="geotags", type=String, filterable=True),
        text("location", filterable=True, facetable=True),
        SimpleField(name="imageurl", type=String, filterable=True),
        SimpleField(name="video_id", type=String, filterable=True, facetable=True),
        text("title", filterable=True),
        text("section", filterable=True),
        SimpleField(name="frame_number", type=SearchFieldDataType.Int32,
                    filterable=True, sortable=True),
        SimpleField(name="prev_id", type=String, filterable=True),
        SimpleField(name="next_id", type=String, filterable=True),
        SimpleField(name="version", type=String, filterable=True),
        SimpleField(name="acl_users", type=StringColl, filterable=True),
        SimpleField(name="acl_groups", type=StringColl, filterable=True),
    ]

    uri = aoai_resource_uri(openai_endpoint)
    vectorizers = []
    if uri:
        vectorizers.append(AzureOpenAIVectorizer(
            vectorizer_name=VECTORIZER_NAME,
            parameters=AzureOpenAIVectorizerParameters(
                resource_url=uri, deployment_name=embedding_deployment,
                model_name=embedding_model, api_key=openai_api_key or None,
            ),
        ))
    vz = VECTORIZER_NAME if uri else None

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=VECTOR_ALGORITHM,
                parameters=HnswParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE,
                    m=4, ef_construction=400, ef_search=500,
                ),
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name=VECTOR_ALGORITHM_EXHAUSTIVE,
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE),
            ),
        ],
        profiles=[
            VectorSearchProfile(name=VECTOR_PROFILE,
                                algorithm_configuration_name=VECTOR_ALGORITHM,
                                vectorizer_name=vz),
            VectorSearchProfile(name=VECTOR_PROFILE_EXHAUSTIVE,
                                algorithm_configuration_name=VECTOR_ALGORITHM_EXHAUSTIVE,
                                vectorizer_name=vz),
        ],
        vectorizers=vectorizers or None,
    )

    semantic = SemanticSearch(
        default_configuration_name=SEMANTIC_CONFIG,
        configurations=[SemanticConfiguration(
            name=SEMANTIC_CONFIG,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name=f)
                                for f in ("caption", "description", "annotations")],
                keywords_fields=[SemanticField(field_name=f)
                                 for f in ("tags", "labels")],
            ),
        )],
    )

    scoring = ScoringProfile(
        name=SCORING_PROFILE,
        text_weights=TextWeights(weights=dict(TEXT_WEIGHTS)),
    )

    return SearchIndex(
        name=name, fields=fields, vector_search=vector_search,
        semantic_search=semantic, similarity=BM25SimilarityAlgorithm(),
        scoring_profiles=[scoring], default_scoring_profile=SCORING_PROFILE,
    )
