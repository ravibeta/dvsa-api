"""Two-step direct upload: presigned destination, completion, storage events, Azure store."""

import datetime
import json
from unittest import mock
from urllib.parse import parse_qs, urlparse

import pytest
from django.test import override_settings
from rest_framework.test import APIClient

from apps.jobs.models import AnalyticsJob
from apps.media.models import MediaAsset
from apps.media import telemetry
from apps.users.models import User
from core.storage.azure_blob import AzureBlobObjectStore, account_key_from_connection_string

FAKE_KEY = "dGVzdGtleXRlc3RrZXl0ZXN0a2V5dGVzdGtleXRlc3RrZXk="  # base64, not a real key
DEST = "/api/v1/media/upload-destination/"
SRT = """1
00:00:00,000 --> 00:00:01,000
[iso : 100] [latitude: 47.6101] [longitude: -122.2015] [rel_alt: 80.5 abs_alt: 120.0]

2
00:00:01,000 --> 00:00:02,000
[latitude: 47.6102] [longitude: -122.2016] [rel_alt: 81.0 abs_alt: 120.5]
"""


def _frames(cfg, url, account_id, video_id=None, stride_seconds=10.0):
    return "stride", [{"frame_number": 0, "blob_name": "k/images/frame0.jpg",
                       "sas_url": "https://x/frame0", "t": 0.0}]


@pytest.fixture
def user(db):
    return User.objects.create_user(email="pilot@example.com", password="pw-123456")


@pytest.fixture
def client(user):
    c = APIClient()
    c.force_authenticate(user=user)
    return c


def _destination(client, **overrides):
    body = {"filename": "Flight 1.MP4", "content_type": "video/mp4", "file_size_bytes": 11,
            "flight_id": "f-1", "metadata": {"gimbal_pitch": -90}}
    body.update(overrides)
    headers = {}
    if "idem" in body:
        headers["HTTP_IDEMPOTENCY_KEY"] = body.pop("idem")
    return client.post(DEST, body, format="json", **headers)


# ------------------------------------------------------------------ step 1: destination
@pytest.mark.django_db
class TestUploadDestination:
    def test_creates_pending_asset_with_write_only_url(self, client, user, fake_store):
        res = _destination(client)
        assert res.status_code == 201, res.content
        body = res.json()
        asset = MediaAsset.objects.get(pk=body["media_id"])
        assert asset.upload_status == "PENDING" and asset.owner == user
        assert body["object_key"] == body["blob_path"] == asset.object_key
        assert asset.object_key == f"{user.pk}/uploads/f-1/{asset.id}/Flight_1.mp4"
        assert body["method"] == "PUT" and body["headers"]["x-ms-blob-type"] == "BlockBlob"
        assert fake_store.upload_requests == [(asset.object_key, "video/mp4", datetime.timedelta(minutes=30))]
        assert asset.storage_provider == "azure" and asset.bucket == "input"
        assert asset.metadata == {"gimbal_pitch": -90}

    def test_same_file_twice_gets_distinct_keys(self, client, fake_store):
        a, b = _destination(client).json(), _destination(client).json()
        assert a["media_id"] != b["media_id"] and a["object_key"] != b["object_key"]

    @pytest.mark.parametrize("overrides", [
        {"filename": "evil.exe", "content_type": "application/octet-stream"},
        {"filename": "clip.mp4", "content_type": "image/png"},
        {"file_size_bytes": 10 ** 15},
        {"filename": "t.srt", "content_type": "text/plain", "file_size_bytes": 60 * 1024 * 1024},
        {"flight_id": "../../etc"},
    ])
    def test_rejects_bad_files(self, client, fake_store, overrides):
        assert _destination(client, **overrides).status_code == 400
        assert MediaAsset.objects.count() == 0

    def test_account_id_must_be_the_caller(self, client, user, fake_store):
        assert _destination(client, account_id=str(user.pk + 1)).status_code == 403
        assert _destination(client, account_id=str(user.pk)).status_code == 201

    def test_requires_authentication(self, fake_store):
        assert APIClient().post(DEST, {}, format="json").status_code == 401

    def test_idempotency_key_returns_same_asset(self, client, fake_store):
        first = _destination(client, idem="abc")
        again = _destination(client, idem="abc")
        assert first.status_code == 201 and again.status_code == 200
        assert first.json()["media_id"] == again.json()["media_id"]
        assert MediaAsset.objects.count() == 1
        assert _destination(client, idem="abc", file_size_bytes=12).status_code == 409

    def test_renew_only_while_pending(self, client, fake_store):
        media_id = _destination(client).json()["media_id"]
        url = f"/api/v1/media/{media_id}/upload-destination/"
        assert client.post(url).status_code == 200
        MediaAsset.objects.filter(pk=media_id).update(upload_status="UPLOADED")
        assert client.post(url).status_code == 409


# ------------------------------------------------------------------- step 2: complete
@pytest.mark.django_db
class TestComplete:
    def _uploaded(self, client, fake_store, data=b"hello world", **overrides):
        body = _destination(client, file_size_bytes=len(data), **overrides).json()
        fake_store.put(body["object_key"], data)
        return body["media_id"]

    def test_missing_object_is_409_and_stays_pending(self, client, fake_store):
        media_id = _destination(client).json()["media_id"]
        res = client.post(f"/api/v1/media/{media_id}/complete/", {}, format="json")
        assert res.status_code == 409
        assert MediaAsset.objects.get(pk=media_id).upload_status == "PENDING"

    def test_size_mismatch_fails_asset(self, client, fake_store):
        body = _destination(client, file_size_bytes=999).json()
        fake_store.put(body["object_key"], b"short")
        res = client.post(f"/api/v1/media/{body['media_id']}/complete/", {}, format="json")
        assert res.status_code == 422
        assert MediaAsset.objects.get(pk=body["media_id"]).upload_status == "FAILED"

    def test_video_completion_dispatches_extract_frames_job(self, client, user, fake_store):
        media_id = self._uploaded(client, fake_store)
        with mock.patch("apps.videos.frame_extract.extract_frames", side_effect=_frames) as fx:
            res = client.post(f"/api/v1/media/{media_id}/complete/", {}, format="json")
        assert res.status_code == 202, res.content
        asset = MediaAsset.objects.get(pk=media_id)
        assert asset.upload_status == "UPLOADED" and asset.etag and asset.uploaded_at
        job = AnalyticsJob.objects.get(pk=res.json()["job_id"])
        assert job.job_type == "EXTRACT_FRAMES" and job.status == "COMPLETED"  # eager Celery
        assert job.attempt == 1 and job.celery_task_id
        kwargs = fx.call_args.kwargs
        assert kwargs["account_id"] == str(user.pk) and kwargs["video_id"] is None
        assert fx.call_args.args[1].endswith(f"{asset.object_key}?sp=r")
        # Repeating the call is idempotent: same job, nothing re-run.
        again = client.post(f"/api/v1/media/{media_id}/complete/", {}, format="json")
        assert again.status_code == 202 and again.json()["job_id"] == str(job.id)
        assert AnalyticsJob.objects.count() == 1

    def test_no_video_entity_is_created(self, client, fake_store):
        from apps.videos.models import VideoEntity

        media_id = self._uploaded(client, fake_store)
        with mock.patch("apps.videos.frame_extract.extract_frames", side_effect=_frames):
            client.post(f"/api/v1/media/{media_id}/complete/", {}, format="json")
        assert VideoEntity.objects.count() == 0  # its post_save would start the indexer

    def test_full_pipeline_rejected_and_asset_untouched(self, client, fake_store):
        media_id = self._uploaded(client, fake_store)
        res = client.post(f"/api/v1/media/{media_id}/complete/", {"job_type": "FULL_PIPELINE"},
                          format="json")
        assert res.status_code == 400 and "per video" in res.json()["error"]
        assert MediaAsset.objects.get(pk=media_id).upload_status == "PENDING"
        assert AnalyticsJob.objects.count() == 0

    def test_explicit_job_type(self, client, fake_store):
        media_id = self._uploaded(client, fake_store)
        with mock.patch("apps.videos.orthomosaic_jobs.run_job",
                        return_value={"state": "completed", "blob": "o.jpg", "output": {}}):
            res = client.post(f"/api/v1/media/{media_id}/complete/",
                              {"job_type": "geospatial_mapping"}, format="json")
        assert res.status_code == 202
        assert res.json()["job"]["job_type"] == "GEOSPATIAL_MAPPING"

    def test_srt_sidecar_captures_telemetry_without_job(self, client, fake_store):
        data = SRT.encode()
        media_id = self._uploaded(client, fake_store, data=data, filename="DJI_0001.SRT",
                                  content_type="application/x-subrip")
        res = client.post(f"/api/v1/media/{media_id}/complete/", {}, format="json")
        assert res.status_code == 200 and "job_id" not in res.json()
        samples = MediaAsset.objects.get(pk=media_id).metadata["telemetry"]
        assert samples[0] == {"t": 0.0, "lat": 47.6101, "lon": -122.2015, "alt": 80.5}
        assert samples[1]["t"] == 1.0
        assert AnalyticsJob.objects.count() == 0

    def test_other_users_asset_is_404(self, client, fake_store):
        media_id = self._uploaded(client, fake_store)
        other = APIClient()
        other.force_authenticate(User.objects.create_user(email="o@example.com", password="pw-123456"))
        assert other.post(f"/api/v1/media/{media_id}/complete/", {}, format="json").status_code == 404
        assert other.get(f"/api/v1/media/{media_id}/").status_code == 404
        assert other.get("/api/v1/media/").json()["count"] == 0

    def test_complete_account_id_mismatch(self, client, user, fake_store):
        media_id = self._uploaded(client, fake_store)
        res = client.post(f"/api/v1/media/{media_id}/complete/", {"account_id": "999999"},
                          format="json")
        assert res.status_code == 403


# ------------------------------------------------------------------- storage events
@pytest.mark.django_db
class TestStorageEvents:
    url = "/api/v1/media/events/azure/"

    def test_disabled_without_token_setting(self, fake_store):
        assert APIClient().post(self.url, [], format="json").status_code == 403

    @override_settings(DVSA_STORAGE_EVENT_TOKEN="s3cret")
    def test_wrong_token(self, fake_store):
        assert APIClient().post(self.url + "?token=nope", [], format="json").status_code == 403

    @override_settings(DVSA_STORAGE_EVENT_TOKEN="s3cret")
    def test_validation_handshake(self, fake_store):
        event = [{"eventType": "Microsoft.EventGrid.SubscriptionValidationEvent",
                  "data": {"validationCode": "code-123"}}]
        res = APIClient().post(self.url + "?token=s3cret", event, format="json")
        assert res.status_code == 200 and res.json() == {"validationResponse": "code-123"}

    @override_settings(DVSA_STORAGE_EVENT_TOKEN="s3cret")
    def test_blob_created_completes_pending_asset(self, client, fake_store):
        body = _destination(client).json()
        fake_store.put(body["object_key"], b"hello world")
        event = [{"eventType": "Microsoft.Storage.BlobCreated",
                  "data": {"url": f"https://acct.blob.core.windows.net/input/{body['object_key']}",
                           "contentLength": 11}},
                 {"eventType": "Microsoft.Storage.BlobCreated",
                  "data": {"url": "https://acct.blob.core.windows.net/input/unknown/key.mp4"}}]
        with mock.patch("apps.videos.frame_extract.extract_frames", side_effect=_frames):
            res = APIClient().post(self.url + "?token=s3cret", event, format="json")
        assert res.status_code == 200
        assert res.json()["received"] == 2 and len(res.json()["completed"]) == 1
        assert MediaAsset.objects.get(pk=body["media_id"]).upload_status == "UPLOADED"
        # The client's own callback afterwards returns the same job.
        again = client.post(f"/api/v1/media/{body['media_id']}/complete/", {}, format="json")
        assert again.json()["job_id"] == res.json()["completed"][0]["job_id"]


# ------------------------------------------------------------------- Azure store
class TestAzureBlobObjectStore:
    store = AzureBlobObjectStore("acct", FAKE_KEY, "input")

    def test_upload_sas_is_create_write_only_for_one_blob(self):
        target = self.store.presign_upload("5/uploads/_/m/clip.mp4", "video/mp4",
                                           datetime.timedelta(minutes=30))
        parsed = urlparse(target.url)
        assert parsed.netloc == "acct.blob.core.windows.net"
        assert parsed.path == "/input/5/uploads/_/m/clip.mp4"
        qs = parse_qs(parsed.query)
        assert qs["sp"] == ["cw"] and qs["sr"] == ["b"] and qs["spr"] == ["https"]
        assert target.method == "PUT" and target.headers["x-ms-blob-type"] == "BlockBlob"
        remaining = target.expires_at - datetime.datetime.now(datetime.timezone.utc)
        assert datetime.timedelta(minutes=29) < remaining <= datetime.timedelta(minutes=30)

    def test_read_sas_is_read_only(self):
        qs = parse_qs(urlparse(self.store.presign_read("a/b.mp4", datetime.timedelta(hours=1))).query)
        assert qs["sp"] == ["r"]

    def test_requires_key_to_sign(self):
        with pytest.raises(RuntimeError):
            AzureBlobObjectStore("acct", None, "input").presign_read("k", datetime.timedelta(hours=1))

    def test_key_from_connection_string(self):
        conn = f"DefaultEndpointsProtocol=https;AccountName=acct;AccountKey={FAKE_KEY};EndpointSuffix=core.windows.net"
        assert account_key_from_connection_string(conn) == FAKE_KEY
        assert AzureBlobObjectStore("acct", None, "input", conn).account_key == FAKE_KEY

    def test_head(self):
        from azure.core.exceptions import ResourceNotFoundError

        props = mock.Mock(size=42, etag='"0x8D"', content_settings=mock.Mock(content_type="video/mp4"))
        client = mock.Mock(get_blob_properties=mock.Mock(return_value=props))
        with mock.patch.object(AzureBlobObjectStore, "_client", return_value=client):
            info = self.store.head("k")
        assert (info.size, info.etag, info.content_type) == (42, "0x8D", "video/mp4")
        client.get_blob_properties.side_effect = ResourceNotFoundError("gone")
        with mock.patch.object(AzureBlobObjectStore, "_client", return_value=client):
            assert self.store.head("k") is None

    def test_event_grid_parsing(self):
        events, handshake = self.store.parse_created_events(json.loads(json.dumps([
            {"eventType": "Microsoft.Storage.BlobCreated",
             "data": {"url": "https://acct.blob.core.windows.net/input/5/uploads/_/m/a%20b.mp4",
                      "contentLength": 7, "eTag": '"0x1"'}},
            {"eventType": "Microsoft.Storage.BlobCreated",
             "data": {"url": "https://other.blob.core.windows.net/input/x.mp4"}},
            {"eventType": "Microsoft.Storage.BlobDeleted", "data": {}},
        ])), {})
        assert handshake is None
        assert [(e.key, e.size, e.etag) for e in events] == [("5/uploads/_/m/a b.mp4", 7, "0x1")]
        _, handshake = self.store.parse_created_events(
            {"eventType": "Microsoft.EventGrid.SubscriptionValidationEvent",
             "data": {"validationCode": "v"}}, {})
        assert handshake == {"validationResponse": "v"}


def test_telemetry_from_metadata():
    assert telemetry.from_metadata({"latitude": "47.1", "longitude": 8.5}) == [
        {"t": 0.0, "lat": 47.1, "lon": 8.5}]
    assert telemetry.from_metadata({"latitude": 0, "longitude": 0}) == []
    assert telemetry.from_metadata({"telemetry": [{"t": 2, "lat": 1, "lon": 2}, {"lat": "x"}]}) == [
        {"t": 2.0, "lat": 1.0, "lon": 2.0}]
    assert telemetry.nearest([{"t": 0, "lat": 1, "lon": 1}, {"t": 5, "lat": 2, "lon": 2}], 4)["lat"] == 2
