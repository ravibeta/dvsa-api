"""Tests for the fabricated-footage screen on the video-upload path.

Fully offline. The pure verdict logic (:func:`core.azure.synth_screen.decide`) is
exercised on synthetic feature dicts, and the endpoint tests patch the seams the
hardened view uses — the MP4 structural validation (``_sniff_mp4_ftyp`` /
``_probe_with_ffprobe``), the screen (:func:`core.azure.synth_screen.screen`),
and the blob upload / registration — so no video runtime, ffprobe, Azure account,
or credentials are needed. They assert a fabricated clip is refused (400) before
it is ever stored or registered, that camera footage passes through to upload
under a server-generated blob name, that oversized / non-MP4 files are rejected,
that screening failures fail open, and that the screen can be disabled entirely
via settings.
"""

from __future__ import annotations

from unittest.mock import patch

from django.conf import settings
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.test import APIClient

from apps.users.models import User
from apps.videos import views as video_views
from apps.videos.views import VideoUploadAPIView, VideoValidationError
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


# ----- module imports without a vision stack -------------------------------
def test_new_features_import_and_do_not_affect_decide():
    # The extra physics-based features are computed but not scored, and the
    # module must import without cv2/scikit-image present.
    assert hasattr(synth_screen, "pixel_distribution_features")
    # decide() ignores every new feature key -> still camera.
    verdict, _cam, _syn = synth_screen.decide({
        "rs_gradient": 0.1, "vignette_slope": -0.2, "demosaic_corr": 0.9,
        "block_periodicity": 3.0, "glcm_contrast": 1.0, "hf_mean": 0.5,
    })
    assert verdict == "camera"


# ----- capped-write helper (streamed size guard) ---------------------------
def test_write_capped_aborts_when_stream_exceeds_cap(tmp_path):
    dest = str(tmp_path / "out.bin")

    class _Fake:
        def chunks(self):
            yield b"x" * 8
            yield b"y" * 8

    try:
        VideoUploadAPIView._write_capped(_Fake(), dest, max_bytes=10)
    except VideoValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected VideoValidationError for oversize stream")


# ----- settings ------------------------------------------------------------
def test_upload_memory_caps_are_bounded():
    # These bound what Django buffers in RAM; they must be small relative to
    # the 500 MB hard cap enforced in the view.
    assert settings.FILE_UPLOAD_MAX_MEMORY_SIZE <= video_views.MAX_UPLOAD_BYTES
    assert settings.DATA_UPLOAD_MAX_MEMORY_SIZE <= video_views.MAX_UPLOAD_BYTES
    assert settings.FILE_UPLOAD_MAX_MEMORY_SIZE > 0
    assert settings.DATA_UPLOAD_MAX_MEMORY_SIZE > 0


def test_synth_screen_enabled_by_default():
    assert getattr(settings, "VIDEO_SYNTH_SCREEN_ENABLED", True) is True


# ----- upload endpoint -----------------------------------------------------
class UploadScreen(TestCase):
    url = "/api/v1/videos/upload-video/"

    def setUp(self):
        cache.clear()  # keep per-scope throttle state from leaking across tests
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="upload@example.com", password="testpass123",
            first_name="Up", last_name="Loader",
        )
        self.client.force_authenticate(user=self.user)

    def _file(self):
        return SimpleUploadedFile("clip.mp4", b"\x00\x01\x02fake-video",
                                  content_type="video/mp4")

    def _valid_mp4(self):
        """Patch the MP4 structural validation to a no-op (a real MP4 would
        require ffprobe); the screen/upload path is what these tests exercise.
        """
        return (
            patch("apps.videos.views._sniff_mp4_ftyp", return_value=None),
            patch("apps.videos.views._probe_with_ffprobe", return_value={}),
        )

    def _post(self):
        return self.client.post(
            self.url, {"account_id": "7", "file": self._file()},
            format="multipart",
        )

    def test_rejects_fabricated_video_before_storing(self):
        sniff, probe = self._valid_mp4()
        with sniff, probe, \
                patch("core.azure.synth_screen.screen",
                      return_value={"verdict": "likely-fabricated"}) as screen, \
                patch("core.azure.blob.service_client") as svc, \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self._post()

        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        body = resp.json()
        assert body["verdict"] == "likely-fabricated"
        assert "camera" in body["error"].lower()
        screen.assert_called_once()
        # Never stored, never registered (so the ingestion signal never fires).
        svc.assert_not_called()
        create.assert_not_called()

    def test_rejects_synthetic_declared_video(self):
        sniff, probe = self._valid_mp4()
        with sniff, probe, \
                patch("core.azure.synth_screen.screen",
                      return_value={"verdict": "synthetic-declared"}), \
                patch("core.azure.blob.service_client") as svc:
            resp = self._post()
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        svc.assert_not_called()

    def test_camera_footage_passes_through_under_server_blob_name(self):
        sniff, probe = self._valid_mp4()
        with sniff, probe, \
                patch("core.azure.synth_screen.screen",
                      return_value={"verdict": "camera"}), \
                patch("core.azure.blob.service_client") as svc, \
                patch("azure.storage.blob.generate_blob_sas", return_value="sig=x"), \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self._post()

        assert resp.status_code == status.HTTP_201_CREATED
        create.assert_called_once()

        # Stored under a server-generated uuid name, never the client filename,
        # and never overwriting an existing blob.
        get_blob = svc.return_value.get_blob_client
        get_blob.assert_called_once()
        blob_name = get_blob.call_args.kwargs["blob"]
        assert blob_name.startswith("7/")
        assert blob_name.endswith(".mp4")
        assert "clip" not in blob_name  # client filename never used as path
        upload_kwargs = get_blob.return_value.upload_blob.call_args.kwargs
        assert upload_kwargs["overwrite"] is False

    def test_review_verdict_passes_through(self):
        sniff, probe = self._valid_mp4()
        with sniff, probe, \
                patch("core.azure.synth_screen.screen",
                      return_value={"verdict": "review"}), \
                patch("core.azure.blob.service_client"), \
                patch("azure.storage.blob.generate_blob_sas", return_value="sig=x"), \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self._post()
        assert resp.status_code == status.HTTP_201_CREATED
        create.assert_called_once()

    def test_screening_failure_fails_open(self):
        # If the screen raises (no ffprobe/cv2, unreadable clip), upload proceeds.
        sniff, probe = self._valid_mp4()
        with sniff, probe, \
                patch("core.azure.synth_screen.screen",
                      side_effect=RuntimeError("no cv2")), \
                patch("core.azure.blob.service_client"), \
                patch("azure.storage.blob.generate_blob_sas", return_value="sig=x"), \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self._post()
        assert resp.status_code == status.HTTP_201_CREATED
        create.assert_called_once()

    @override_settings(VIDEO_SYNTH_SCREEN_ENABLED=False)
    def test_screen_can_be_disabled_via_settings(self):
        # With screening off, the screen is never consulted and the upload
        # proceeds even for what would be a fabricated verdict.
        sniff, probe = self._valid_mp4()
        with sniff, probe, \
                patch("core.azure.synth_screen.screen") as screen, \
                patch("core.azure.blob.service_client"), \
                patch("azure.storage.blob.generate_blob_sas", return_value="sig=x"), \
                patch("apps.videos.models.VideoEntity.create_video") as create:
            resp = self._post()
        assert resp.status_code == status.HTTP_201_CREATED
        screen.assert_not_called()
        create.assert_called_once()

    def test_rejects_non_mp4_content(self):
        # No validation patch: the real ftyp sniff rejects renamed non-MP4 bytes
        # before any screening or storage happens.
        with patch("core.azure.synth_screen.screen") as screen, \
                patch("core.azure.blob.service_client") as svc:
            resp = self._post()
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "mp4" in resp.json()["error"].lower()
        screen.assert_not_called()
        svc.assert_not_called()

    def test_rejects_oversized_file(self):
        # Shrink the cap so the declared-size guard trips on a tiny upload.
        with patch("apps.videos.views.MAX_UPLOAD_BYTES", 4), \
                patch("core.azure.blob.service_client") as svc:
            resp = self._post()
        assert resp.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        svc.assert_not_called()

    def test_rejects_invalid_account_id(self):
        sniff, probe = self._valid_mp4()
        with sniff, probe:
            resp = self.client.post(
                self.url, {"account_id": "abc", "file": self._file()},
                format="multipart",
            )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

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
