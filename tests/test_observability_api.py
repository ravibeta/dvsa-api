"""Database/API tests for the observability layer.

Unlike the other observability test modules (which are pure-Python), these
exercise the full Django stack: the ``CommentaryEventRecord`` model, the DRF
ingest/list/aggregate endpoints and the semantic-agent endpoint. They require
the DB, so they are marked ``django_db`` and run under pytest-django.
"""

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from apps.observability.models import CommentaryEventRecord

pytestmark = pytest.mark.django_db

BASE = "/api/v1/observability"


@pytest.fixture
def client():
    user = get_user_model().objects.create_user(email="tester@example.com", password="pw-123456789")
    api = APIClient()
    api.force_authenticate(user)
    return api


def _ingest(client, **payload):
    resp = client.post("%s/events/" % BASE, payload, format="json")
    assert resp.status_code == 201, resp.content
    return resp.json()


# --------------------------------------------------------------------------- #
# Ingest (custom event injection) + persistence
# --------------------------------------------------------------------------- #

def test_ingest_creates_persisted_event(client):
    data = _ingest(
        client,
        commentary="clear to land",
        attributes={"verdict": "safe"},
        metrics={"confidence": 0.9},
        video_id=5,
        frame_index=2,
    )
    assert data["commentary"] == "clear to land"
    assert data["attributes"]["verdict"] == "safe"
    assert len(data["trace_id"]) == 32 and len(data["span_id"]) == 16
    assert data["source"] == "external"

    rec = CommentaryEventRecord.objects.get(event_id=data["event_id"])
    assert rec.video_id == 5 and rec.frame_index == 2
    assert rec.metrics["confidence"] == 0.9


def test_list_filters_by_video_and_source(client):
    _ingest(client, commentary="a", source="routine:color_detection", video_id=1)
    _ingest(client, commentary="b", source="agent:vlm", video_id=1)
    _ingest(client, commentary="c", source="routine:color_detection", video_id=2)

    resp = client.get("%s/events/?video_id=1" % BASE)
    assert resp.status_code == 200
    assert resp.json()["count"] == 2

    resp = client.get("%s/events/?source=agent:vlm" % BASE)
    assert resp.json()["count"] == 1


# --------------------------------------------------------------------------- #
# Query-time aggregation endpoint
# --------------------------------------------------------------------------- #

def test_aggregate_endpoint_groups_and_reduces(client):
    _ingest(client, commentary="1", source="routine:color_detection", video_id=9, metrics={"count": 2.0})
    _ingest(client, commentary="2", source="routine:color_detection", video_id=9, metrics={"count": 3.0})
    _ingest(client, commentary="3", source="agent:vlm", video_id=9, metrics={"count": 1.0})

    resp = client.get("%s/events/aggregate/?video_id=9&group_by=source&metrics=count,sum:count" % BASE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_events"] == 3
    cd = body["groups"]["routine:color_detection"]
    assert cd["events"] == 2
    assert cd["sum:count"] == 5.0


def test_aggregate_rejects_bad_metric_op(client):
    _ingest(client, commentary="x", video_id=1)
    resp = client.get("%s/events/aggregate/?metrics=bogus:count" % BASE)
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Semantic agent endpoint
# --------------------------------------------------------------------------- #

def test_agent_summarize_creates_higher_level_event(client):
    _ingest(client, commentary="1 car", source="routine:color_detection",
            video_id=42, frame_index=0, metrics={"count": 1.0})
    _ingest(client, commentary="2 car", source="routine:color_detection",
            video_id=42, frame_index=30, metrics={"count": 2.0})

    resp = client.post("%s/agents/summarize/" % BASE, {"video_id": 42, "scope": "video"}, format="json")
    assert resp.status_code == 201, resp.content
    body = resp.json()
    assert body["summarized"] == 2
    assert len(body["events"]) == 1

    summary = CommentaryEventRecord.objects.get(source="agent:semantic")
    assert summary.video_id == 42
    assert summary.correlation_key == "video:42|scope:video"
    assert summary.metrics["events_summarized"] == 2.0
    # The summary must not summarise itself on a second run.
    resp2 = client.post("%s/agents/summarize/" % BASE, {"video_id": 42}, format="json")
    assert resp2.json()["summarized"] == 2


def test_agent_summarize_requires_a_selector(client):
    resp = client.post("%s/agents/summarize/" % BASE, {}, format="json")
    assert resp.status_code == 400


def test_agent_summarize_no_events_returns_200(client):
    resp = client.post("%s/agents/summarize/" % BASE, {"video_id": 999}, format="json")
    assert resp.status_code == 200
    assert resp.json()["summarized"] == 0
