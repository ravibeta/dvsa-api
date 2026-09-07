"""Tests for the ``/corners`` endpoint and its corner-extraction runner.

Fully offline. The pure geometry (corner ordering / rectangle sampling) is
exercised on synthetic tracks, and the endpoint tests patch the three seams the
view uses — the blob download, :func:`core.azure.survey_corners.extract_corner_frames`,
and the upload/SAS mint — so no video runtime, Azure account, or credentials are
needed. They assert the endpoint bypasses the agentic path, writes each corner to
``{account_id}/images/{video_pk}/corners/{label}.jpg``, and returns a downloadable
SAS URL per corner (BL -> TL -> TR -> BR).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from apps.users.models import User
from apps.videos.models import VideoEntity
from core.azure import survey_corners


# ----- pure geometry -------------------------------------------------------
def test_order_clockwise_from_bottom_left():
    # y-up square: BL, TL, TR, BR
    box = [(0, 0), (0, 10), (10, 10), (10, 0)]
    ordered = survey_corners.order_clockwise_from_bottom_left(box, np.array([5, 5]))
    assert [tuple(p) for p in ordered] == [(0, 0), (0, 10), (10, 10), (10, 0)]


def test_corner_samples_orders_bl_clockwise():
    # Dense points along the edges of an axis-aligned rectangle.
    edge = np.linspace(0, 10, 11)
    pts = []
    for t in edge:
        pts += [(t, 0), (10, t), (t, 10), (0, t)]
    track = np.array(pts, dtype=float)

    picks = survey_corners.corner_samples(track)
    assert len(picks) == 4
    corners = track[picks]
    # BL -> TL -> TR -> BR in the y-up frame.
    assert np.allclose(corners[0], (0, 0))
    assert np.allclose(corners[1], (0, 10))
    assert np.allclose(corners[2], (10, 10))
    assert np.allclose(corners[3], (10, 0))


# ----- endpoint ------------------------------------------------------------
def _fake_frames():
    return [
        ("1-bottom-left", b"\xff\xd8bl", {"frame": 10, "t": 0.5, "x": 0.0, "y": 0.0,
                                          "mode": "visual"}),
        ("2-top-left", b"\xff\xd8tl", {"frame": 20, "t": 1.0, "x": 0.0, "y": 9.0,
                                       "mode": "visual"}),
        ("3-top-right", b"\xff\xd8tr", {"frame": 30, "t": 1.5, "x": 9.0, "y": 9.0,
                                        "mode": "visual"}),
        ("4-bottom-right", b"\xff\xd8br", {"frame": 40, "t": 2.0, "x": 9.0, "y": 0.0,
                                           "mode": "visual"}),
    ]


def _fake_sas(cfg, blob_name, data, **kwargs):
    return f"https://sadronevideo.blob.core.windows.net/input/{blob_name}?sig=x"


class CornersEndpoint(TestCase):
    url = "/api/v1/videos/corners/"

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="corners@example.com", password="testpass123",
            first_name="Cor", last_name="Ners",
        )

    def _patches(self):
        return (
            patch("core.azure.blob.download_blob_to_temp", return_value="/nope/x.mp4"),
            patch("core.azure.survey_corners.extract_corner_frames",
                  return_value=_fake_frames()),
            patch("core.azure.blob.put_blob_and_sas", side_effect=_fake_sas),
        )

    def test_returns_four_corner_sas_urls_for_latest_video(self):
        self.client.force_authenticate(user=self.user)
        entity = VideoEntity()
        entity.create_video(
            account_id="7",
            sas_url="https://sadronevideo.blob.core.windows.net/input/7/tour.mp4",
        )
        dl, extract, sas = self._patches()
        with dl as dl_m, extract, sas:
            resp = self.client.put(self.url, {"account_id": "7"})

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert body["video_pk"] == entity.pk
        assert body["count"] == 4
        labels = [c["label"] for c in body["corners"]]
        assert labels == ["1-bottom-left", "2-top-left", "3-top-right", "4-bottom-right"]
        # Written next to the video's frames under the corners folder.
        assert (f"images/{entity.pk}/corners/1-bottom-left.jpg"
                in body["corners"][0]["downloadUrl"])
        # Source video came from the resolved entity's sas_url.
        dl_m.assert_called_once_with(entity.sas_url)

    def test_explicit_sas_url_defaults_video_pk_to_zero(self):
        self.client.force_authenticate(user=self.user)
        dl, extract, sas = self._patches()
        with dl, extract, sas:
            resp = self.client.put(self.url, {
                "account_id": "9",
                "sas_url": "https://sadronevideo.blob.core.windows.net/input/9/x.mp4",
            })
        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert body["video_pk"] == 0
        assert "images/0/corners/1-bottom-left.jpg" in body["corners"][0]["downloadUrl"]

    def test_video_id_not_found_returns_404(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.put(self.url, {"account_id": "7", "video_id": "999999"})
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    def test_no_video_for_account_returns_404(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.put(self.url, {"account_id": "does-not-exist"})
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    def test_requires_account_id(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.put(self.url, {})
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_requires_authentication(self):
        resp = self.client.put(self.url, {"account_id": "7"})
        assert resp.status_code in (status.HTTP_401_UNAUTHORIZED,
                                    status.HTTP_403_FORBIDDEN)
