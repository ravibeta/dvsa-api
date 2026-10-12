"""Index schema: retrieval capabilities declared on every DVSA index.

Covers the Vector-Search-Index-Retrieval-Settings pipeline at the index level:
dense (HNSW + exhaustive KNN) + lexical (BM25) + semantic rerank + context
assembly + version/ACL fields.
"""

import json

import pytest

from core.azure import index_schema

ENDPOINT = "https://found-vision-1.openai.azure.com/openai/v1"


def _names(desc):
    return {f["name"] for f in desc["fields"]}


def test_required_capability_fields_present():
    names = _names(index_schema.describe_index("idx"))
    for f in ("vector", "caption", "location", "timestamp", "tags", "labels",
              "annotations", "video_id", "title", "section", "frame_number",
              "prev_id", "next_id", "version", "acl_users", "acl_groups"):
        assert f in names, f


def test_both_vector_algorithms_and_profiles_described():
    vs = index_schema.describe_index("idx")["vector_search"]
    assert {a["kind"] for a in vs["algorithms"]} == {"hnsw", "exhaustiveKnn"}
    assert {p["name"] for p in vs["profiles"]} == {
        index_schema.VECTOR_PROFILE, index_schema.VECTOR_PROFILE_EXHAUSTIVE}
    assert "vectorizers" not in vs  # no endpoint -> no vectorizer


def test_vectorizer_strips_unified_openai_v1_suffix():
    vs = index_schema.describe_index("idx", openai_endpoint=ENDPOINT)["vector_search"]
    assert vs["vectorizers"][0]["resource_uri"] == "https://found-vision-1.openai.azure.com"
    assert all(p["vectorizer"] == index_schema.VECTOR_VECTORIZER
               if hasattr(index_schema, "VECTOR_VECTORIZER")
               else p["vectorizer"] == index_schema.VECTORIZER_NAME
               for p in vs["profiles"])


def test_semantic_and_scoring_declared():
    d = index_schema.describe_index("idx")
    assert d["semantic"]["default_configuration"] == index_schema.SEMANTIC_CONFIG
    assert d["scoring_profile"]["text_weights"]["caption"] > d["scoring_profile"]["text_weights"]["description"]


def test_built_search_index_serializes_with_all_capabilities():
    pytest.importorskip("azure.search.documents")
    idx = index_schema.build_search_index(
        "dvsa-index-s-account-5", 1536, openai_endpoint=ENDPOINT,
        openai_api_key=None)
    body = json.loads(json.dumps(idx.serialize()))
    kinds = {a["kind"] for a in body["vectorSearch"]["algorithms"]}
    assert kinds == {"hnsw", "exhaustiveKnn"}
    assert {p["name"] for p in body["vectorSearch"]["profiles"]} == {
        "dvsa-hnsw-profile", "dvsa-eknn-profile"}
    assert body["vectorSearch"]["vectorizers"][0]["azureOpenAIParameters"]["resourceUri"] \
        == "https://found-vision-1.openai.azure.com"
    assert body["semantic"]["defaultConfiguration"] == "dvsa-semantic"
    assert body["defaultScoringProfile"] == "dvsa-boost"
    vec = next(f for f in body["fields"] if f["name"] == "vector")
    assert vec["dimensions"] == 1536
    assert vec["vectorSearchProfile"] == "dvsa-hnsw-profile"
    assert {f["name"] for f in body["fields"]} >= _names(index_schema.describe_index("x"))


def test_built_index_without_endpoint_has_no_vectorizer():
    pytest.importorskip("azure.search.documents")
    body = json.loads(json.dumps(index_schema.build_search_index("i", 8).serialize()))
    assert not body["vectorSearch"].get("vectorizers")
