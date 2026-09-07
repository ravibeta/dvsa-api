"""Tests for the fabricated-footage screen on the video-upload path.

Fully offline. The pure verdict logic (:func:`core.azure.synth_screen.decide`) is
exercised on synthetic feature dicts, and the endpoint tests patch the one seam
the view uses — :func:`core.azure.synth_screen.screen` — plus the blob upload /
registration, so no video runtime, ffprobe, Azure account, or credentials are
needed. They assert a fabricated clip is refused (400) before it is ever stored
or registered, that camera footage passes through to upload, and that screening
failures fail open (the upload proceeds).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from apps.users.models import User
from core.azure import synth_screen


# ----- pure verdict logic --------------------------------------------------
def test_default_verdict_is_camera_on_empty_features():
    # Absence of evidence is never scored -> defaults to camera.
    verdict, cam, syn = synth_screen.decide({})
    assert verdict == "camera"
    assert syn == []


def test_generator_tag_is_synthetic_declared():
    verdict, _cam, _syn = synth_screen.decide({"generator_tag": True})
    assert verdict == "synthetic-declared"


def test_c2pa_ai_manifest_is_synthetic_declared():
    verdict, _cam, _syn = synth_screen.decide({"c2pa_ai_manifest": True})
    assert verdict == "synthetic-declared"


def test_three_signs_no_camera_evidence_is_likely_fabricated():
    verdict, cam, syn = synth_screen.decide({
        "residual_corr": 0.001,     # no_sensor_pattern
        "residual_energy": 0.5,     # too_little_grain
        "inlier_ratio": 0.5,        # incoherent_scene
    })
    assert verdict == "likely-fabricated"
    assert cam == []
    assert len(syn) >= synth_screen.MIN_SYNTH_SIGNS


def test_fewer_signs_is_review():
    verdict, _cam, syn = synth_screen.decide({"inlier_ratio": 0.5})
    assert verdict == "review"
    assert syn == ["incoherent_scene"]


def test_sensor_pattern_vetoes_fabrication_signs():
    # Even with fabrication signs, veto-grade camera evidence lands on camera.
    verdict, cam, _syn = synth_screen.decide({
        "residual_corr": 0.05,      # >= CAMERA_EVIDENCE -> sensor_pattern_noise
        "residual_energy": 0.5,
        "inlier_ratio": 0.5,
        "reproj_error": 2.0,
    })
    assert verdict == "camera"
    assert "sensor_pattern_noise" in cam


def test_camera_metadata_vetoes():
    verdict, cam, _syn = synth_screen.decide({
        "has_camera_metadata": True,
        "residual_energy": 0.5,
        "inlier_ratio": 0.5,
        "reproj_error": 2.0,
    })
    assert verdict == "camera"
    assert "camera_provenance" in cam


def test_fabricated_verdicts_set():
    assert synth_screen.FABRICATED_VERDICTS == frozenset(
        {"likely-fabricated", "synthetic-declared"}
    )


# ----- upload endpoint -----------------------------------------------------
class UploadScreen(TestCase):
    url = "/api/v1/videos/upload-video/"

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="upload@example.com", password="testpass123",
            first_name="Up", last_name="Loader",
        )
        self.client.force_authenticate(user=self.user)

    def _file(self):
        return SimpleUploadedFile("clip.mp4", b"\x00\x01\x02fake-video",
                                  content_type="video/mp4")

    def test_rejects_fabricated_video_before_storing(self):
        with patch("core.azure.synth_screen.screen",
                   return_value={"verdict": "likely-fabricated"}) as screen, \
                patch("core.azure.blob.service_client") as svc, \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self.client.post(
                self.url, {"account_id": "7", "file": self._file()},
                format="multipart",
            )

        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        body = resp.json()
        assert body["verdict"] == "likely-fabricated"
        assert "camera" in body["error"].lower()
        screen.assert_called_once()
        # Never stored, never registered (so the ingestion signal never fires).
        svc.assert_not_called()
        create.assert_not_called()

    def test_rejects_synthetic_declared_video(self):
        with patch("core.azure.synth_screen.screen",
                   return_value={"verdict": "synthetic-declared"}), \
                patch("core.azure.blob.service_client") as svc:
            resp = self.client.post(
                self.url, {"account_id": "7", "file": self._file()},
                format="multipart",
            )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        svc.assert_not_called()

    def test_camera_footage_passes_through_to_upload(self):
        with patch("core.azure.synth_screen.screen",
                   return_value={"verdict": "camera"}), \
                patch("core.azure.blob.service_client", return_value=MagicMock()), \
                patch("azure.storage.blob.generate_blob_sas", return_value="sig=x"), \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self.client.post(
                self.url, {"account_id": "7", "file": self._file()},
                format="multipart",
            )
        assert resp.status_code == status.HTTP_201_CREATED
        create.assert_called_once()

    def test_review_verdict_passes_through(self):
        with patch("core.azure.synth_screen.screen",
                   return_value={"verdict": "review"}), \
                patch("core.azure.blob.service_client", return_value=MagicMock()), \
                patch("azure.storage.blob.generate_blob_sas", return_value="sig=x"), \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self.client.post(
                self.url, {"account_id": "7", "file": self._file()},
                format="multipart",
            )
        assert resp.status_code == status.HTTP_201_CREATED
        create.assert_called_once()

    def test_screening_failure_fails_open(self):
        # If the screen raises (no ffprobe/cv2, unreadable clip), upload proceeds.
        with patch("core.azure.synth_screen.screen",
                   side_effect=RuntimeError("no cv2")), \
                patch("core.azure.blob.service_client", return_value=MagicMock()), \
                patch("azure.storage.blob.generate_blob_sas", return_value="sig=x"), \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self.client.post(
                self.url, {"account_id": "7", "file": self._file()},
                format="multipart",
            )
        assert resp.status_code == status.HTTP_201_CREATED
        create.assert_called_once()

    def test_requires_file_and_account_id(self):
        resp = self.client.post(self.url, {"account_id": "7"}, format="multipart")
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_requires_authentication(self):
        self.client.force_authenticate(user=None)
        resp = self.client.post(
            self.url, {"account_id": "7", "file": self._file()}, format="multipart",
        )
        assert resp.status_code in (status.HTTP_401_UNAUTHORIZED,
                                    status.HTTP_403_FORBIDDEN)
