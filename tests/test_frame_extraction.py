"""Tests for the ``/extract-frames/`` endpoint and its extraction runner.

Fully offline, same style as ``tests/test_corners.py``: the blob download,
frame sampling, and SAS mint are patched, so no video runtime, Azure account,
or credentials are needed. They assert the endpoint writes sequential
``{account_id}/images/{video_pk}/frame{N}.jpg`` blobs into the shared ingestion
layout, is idempotent, paginates, and does not disturb the ingestion skip.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from apps.users.models import User
from apps.videos.models import VideoEntity

_JPEG = b"\xff\xd8jpg"
_SAS = "https://sadronevideo.blob.core.windows.net/input/7/tour.mp4"


def _fake_put(cfg, blob_name, data, **kwargs):
    return f"https://sadronevideo.blob.core.windows.net/input/{blob_name}?sig=put"


def _fake_read(cfg, blob_name, **kwargs):
    return f"https://sadronevideo.blob.core.windows.net/input/{blob_name}?sig=read"


class FrameExtractEndpoint(TestCase):
    url = "/api/v1/videos/extract-frames/"

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="frames@example.com", password="testpass123",
            first_name="Fra", last_name="Mes",
        )
        self.entity = VideoEntity()
        self.entity.create_video(account_id="7", sas_url=_SAS)

    def test_stride_happy_path_writes_sequential_frames(self):
        self.client.force_authenticate(user=self.user)
        samples = [(0.0, _JPEG), (10.0, _JPEG), (20.0, _JPEG)]
        with patch("core.azure.blob.get_uploaded_frames", return_value=0), \
             patch("core.azure.blob.download_blob_to_temp", return_value="/nope/x.mp4") as dl, \
             patch("apps.videos.frame_extract._video_indexer_samples", return_value=None), \
             patch("apps.videos.frame_extract._stride_samples", return_value=samples) as strider, \
             patch("core.azure.blob.put_blob_and_sas", side_effect=_fake_put) as put:
            resp = self.client.post(self.url, {"account_id": "7", "stride": 10})

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert body["source"] == "stride"
        assert body["video_pk"] == self.entity.pk
        assert body["count"] == 3
        names = [r["blob_name"] for r in body["results"]]
        assert names == [
            f"7/images/{self.entity.pk}/frame0.jpg",
            f"7/images/{self.entity.pk}/frame1.jpg",
            f"7/images/{self.entity.pk}/frame2.jpg",
        ]
        assert [r["frame_number"] for r in body["results"]] == [0, 1, 2]
        assert put.call_count == 3
        dl.assert_called_once_with(self.entity.sas_url)
        strider.assert_called_once()
        # ImageEntity rows persisted for the entity.
        assert self.entity.images.count() == 3

    def test_idempotent_second_call_does_not_reextract(self):
        self.client.force_authenticate(user=self.user)
        with patch("core.azure.blob.get_uploaded_frames", return_value=3), \
             patch("core.azure.blob.download_blob_to_temp") as dl, \
             patch("apps.videos.frame_extract._stride_samples") as strider, \
             patch("core.azure.blob.read_sas_for_blob", side_effect=_fake_read):
            resp = self.client.post(self.url, {"account_id": "7"})

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert body["source"] == "existing"
        assert body["count"] == 3
        assert [r["frame_number"] for r in body["results"]] == [0, 1, 2]
        assert "sig=read" in body["results"][0]["sas_url"]
        dl.assert_not_called()
        strider.assert_not_called()

    def test_explicit_sas_url_omits_video_segment(self):
        self.client.force_authenticate(user=self.user)
        with patch("core.azure.blob.get_uploaded_frames", return_value=0), \
             patch("core.azure.blob.download_blob_to_temp", return_value="/nope/x.mp4"), \
             patch("apps.videos.frame_extract._video_indexer_samples", return_value=None), \
             patch("apps.videos.frame_extract._stride_samples", return_value=[(0.0, _JPEG)]), \
             patch("core.azure.blob.put_blob_and_sas", side_effect=_fake_put):
            resp = self.client.post(self.url, {
                "account_id": "9",
                "video_sas_url": "https://sadronevideo.blob.core.windows.net/input/9/x.mp4",
            })
        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert body["video_pk"] == 0
        # No entity -> no {video_id} path segment.
        assert body["results"][0]["blob_name"] == "9/images/frame0.jpg"

    def test_pagination_page_size_and_next_link(self):
        self.client.force_authenticate(user=self.user)
        samples = [(float(i), _JPEG) for i in range(3)]
        with patch("core.azure.blob.get_uploaded_frames", return_value=0), \
             patch("core.azure.blob.download_blob_to_temp", return_value="/nope/x.mp4"), \
             patch("apps.videos.frame_extract._video_indexer_samples", return_value=None), \
             patch("apps.videos.frame_extract._stride_samples", return_value=samples), \
             patch("core.azure.blob.put_blob_and_sas", side_effect=_fake_put):
            resp = self.client.post(self.url + "?page_size=2&page=1", {"account_id": "7"})

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert body["count"] == 3
        assert len(body["results"]) == 2
        assert body["next"] is not None

    def test_requires_account_id(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.post(self.url, {})
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_no_video_for_account_returns_404(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.post(self.url, {"account_id": "does-not-exist"})
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    def test_video_id_not_found_returns_404(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.post(self.url, {"account_id": "7", "video_id": 999999})
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    def test_requires_authentication(self):
        resp = self.client.post(self.url, {"account_id": "7"})
        assert resp.status_code in (status.HTTP_401_UNAUTHORIZED,
                                    status.HTTP_403_FORBIDDEN)


# ----- stride sampler unit (fake OpenCV) -----------------------------------
def _fake_cv2():
    fps_key, count_key, msec_key = 5, 7, 0

    class FakeCap:
        def __init__(self):
            self.msec = []

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == fps_key:
                return 30.0
            if prop == count_key:
                return 900  # 30 s at 30 fps
            return 0

        def set(self, prop, val):
            if prop == msec_key:
                self.msec.append(val)

        def read(self):
            return True, "frame"

        def release(self):
            pass

    class FakeBuf:
        def tobytes(self):
            return _JPEG

    cap = FakeCap()
    return SimpleNamespace(
        CAP_PROP_FPS=fps_key, CAP_PROP_FRAME_COUNT=count_key, CAP_PROP_POS_MSEC=msec_key,
        VideoCapture=lambda _p: cap,
        imencode=lambda _ext, _frame: (True, FakeBuf()),
        _cap=cap,
    )


def test_stride_sampler_seeks_at_time_stride():
    from apps.videos import frame_extract

    fake = _fake_cv2()
    with patch.dict(sys.modules, {"cv2": fake}):
        samples = frame_extract._stride_samples("/x.mp4", 10.0)

    # 0,10,20,30 s (<= 30 s duration).
    assert fake._cap.msec == [0.0, 10000.0, 20000.0, 30000.0]
    assert [s[0] for s in samples] == [0.0, 10.0, 20.0, 30.0]
    assert all(s[1] == _JPEG for s in samples)


# ----- ingestion skip ------------------------------------------------------
def test_ingest_video_skips_reextract_when_frames_exist():
    from core.azure import create_session_azure_environment
    from core.azure.config import AzureEnvironmentConfig

    env = create_session_azure_environment(
        "sess-skip", config=AzureEnvironmentConfig(provisioner="dryrun"))
    with patch("core.azure.blob.get_uploaded_frames", return_value=5), \
         patch("core.azure.blob.extract_and_upload_frames") as extract, \
         patch.object(type(env), "vectorize_and_index_frame", return_value={}):
        result = env.ingest_video("https://sadronevideo.blob.core.windows.net/input/7/tour.mp4",
                                  account_id="7", video_id=1)

    assert result["frames"] == 5
    extract.assert_not_called()
