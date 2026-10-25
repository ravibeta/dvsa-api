"""Tests for the orthomosaic pipeline port, its background job, and the ``/orthomosaic/`` endpoint.

Fully offline: Docker / ODM / Azure Storage are never touched. The pure parts of the ported
``generate_orthomosaic.py`` are exercised directly (options, ODM/renderer commands, render
planning, OBJ inspection, frame extraction on a tiny synthetic video); the job and endpoint tests
patch the seams (engine check, blob download/upload, ``build_orthomosaic``) and assert the key
guarantees: the result lands at ``{account}/images/{video}/orthomosaic.jpg``, a missing
Docker/ODM install is a clean ``failed`` state, and nothing raises into the upload path.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from django.core.cache import cache
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.test import APIClient

from apps.users.models import User
from apps.videos import orthomosaic_jobs as jobs
from apps.videos.models import VideoEntity
from core.azure import orthomosaic as ortho

SAS = "https://sadronevideo.blob.core.windows.net/input/7/tour.mp4?sig=x"


# ----- pure pipeline pieces ---------------------------------------------------------------
def test_blob_name_uses_the_shared_frame_layout():
    assert ortho.orthomosaic_blob_name("7", 42) == "7/images/42/orthomosaic.jpg"


def test_positive_rejects_nonfinite_and_nonpositive():
    ortho.positive(1, "x")
    ortho.positive(0, "x", zero=True)
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ortho.positive(bad, "x")


def test_reconstruction_options_match_the_script_and_fast_flag():
    opts = ortho.reconstruction_options(False, 4, "high", "medium", 12)
    assert opts[:2] == ["--feature-type", "sift"] and "--skip-3dmodel" in opts
    assert opts[opts.index("--end-with") + 1] == "mvs_texturing"
    assert "--fast-orthophoto" not in opts
    assert "--fast-orthophoto" in ortho.reconstruction_options(True, 4, "high", "medium", 12)
    with pytest.raises(ValueError):
        ortho.reconstruction_options(False, 0, "high", "medium", 12)


def test_odm_command_docker_and_native(tmp_path):
    project = tmp_path / "project"
    docker = {"backend": "docker", "docker": "docker", "image_id": "sha256:abc"}
    cmd = ortho.odm_command(project, docker, ["--fast-orthophoto"], rerun=True)
    assert cmd[:5] == ["docker", "run", "--rm", "--pull=never", "-v"]
    assert cmd[5] == f"{tmp_path}:/datasets" and "sha256:abc" in cmd
    assert cmd[-2:] == ["--fast-orthophoto", "--rerun-all"] and "/datasets" in cmd

    native = {"backend": "native", "python": "py", "run_py": "/odm/run.py"}
    cmd = ortho.odm_command(project, native, [])
    assert cmd[:2] == ["py", "/odm/run.py"] and str(tmp_path) in cmd


def test_resolve_engine_requires_docker_or_a_native_run_py(tmp_path):
    with patch("core.azure.orthomosaic.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Docker not found"):
            ortho.resolve_engine("docker", "img", None, None)
    with pytest.raises(RuntimeError, match="--odm-run"):
        ortho.resolve_engine("native", "img", None, None)
    run = tmp_path / "run.py"
    run.write_text("print()")
    engine = ortho.resolve_engine("native", "img", str(run), "python3")
    assert engine["backend"] == "native" and engine["python"] == "python3" and engine["run_sha256"]


def test_resolve_engine_inspects_the_local_image():
    ok = SimpleNamespace(returncode=0, stdout="sha256:feed\n", stderr="")
    with patch("core.azure.orthomosaic.shutil.which", return_value="docker"), \
            patch("core.azure.orthomosaic.subprocess.run", return_value=ok):
        assert ortho.resolve_engine("docker", "img", None, None)["image_id"] == "sha256:feed"
    missing = SimpleNamespace(returncode=1, stdout="", stderr="no such image")
    with patch("core.azure.orthomosaic.shutil.which", return_value="docker"), \
            patch("core.azure.orthomosaic.subprocess.run", return_value=missing):
        with pytest.raises(RuntimeError, match="Cannot inspect local ODM image"):
            ortho.resolve_engine("docker", "img", None, None)


def test_render_plan_density_and_limits():
    info = {"span_xy": [10.0, 5.0]}
    plan = ortho.render_plan(info, 6000)
    assert plan["estimated_width"] == 6000 and plan["estimated_height"] == 3000
    assert plan["pixels_per_model_unit"] == pytest.approx(600)
    with pytest.raises(ValueError, match="between 256 and 20000"):
        ortho.render_plan(info, 100)
    with pytest.raises(ValueError, match="over --max-megapixels"):
        ortho.render_plan(info, 20000, max_megapixels=1)
    with pytest.raises(ValueError, match="Invalid XY mesh extent"):
        ortho.render_plan({"span_xy": [0, 5]}, 6000)


def test_renderer_command_maps_paths_into_the_container(tmp_path):
    project = tmp_path / "project"
    model = project / "odm_texturing_25d" / "m.obj"
    tif = project / "odm_orthophoto" / "o.tif"
    docker = {"backend": "docker", "docker": "docker", "image_id": "sha256:abc"}
    cmd = ortho.renderer_command(project, model, tif, {"pixels_per_model_unit": 12.5}, docker, 4)
    assert "--entrypoint" in cmd and "/datasets/project/odm_texturing_25d/m.obj" in cmd
    assert "/datasets/project/odm_orthophoto/o.tif" in cmd and "12.5" in cmd


def _write_model(folder: Path, *, texture=True, uv=True):
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "m.mtl").write_text("newmtl a\nmap_Kd tex.jpg\n")
    if texture:
        Image.new("RGB", (4, 4), (10, 20, 30)).save(folder / "tex.jpg")
    obj = ["mtllib m.mtl", "v 0 0 0", "v 10 0 0", "v 10 5 1", "v 0 5 1"]
    if uv:
        obj += ["vt 0 0", "vt 1 0", "vt 1 1"]
    obj += ["f 1/1 2/2 3/3", "f 1/1 3/3 4/3"]
    (folder / "m.obj").write_text("\n".join(obj))
    return folder / "m.obj"


def test_inspect_model_reports_extent_and_assets(tmp_path):
    info = ortho.inspect_model(_write_model(tmp_path / "model"))
    assert info["span_xy"] == [10.0, 5.0] and info["vertices"] == 4 and info["faces"] == 2
    assert [p.name for p in info["textures"]] == ["tex.jpg"]
    sig = ortho.artifact_signature(info, tmp_path)
    assert {s["path"] for s in sig} == {"model/m.obj", "model/m.mtl", "model/tex.jpg"}


def test_inspect_model_rejects_incomplete_meshes(tmp_path):
    with pytest.raises(ValueError, match="lacks vertices, faces or texture"):
        ortho.inspect_model(_write_model(tmp_path / "a", uv=False))
    with pytest.raises(ValueError, match="Missing texture"):
        ortho.inspect_model(_write_model(tmp_path / "b", texture=False))


def test_extract_frames_samples_a_synthetic_video_and_reuses_the_verified_cache(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 here")
    rng = np.random.default_rng(1)
    for _ in range(35):
        writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()

    frames = ortho.extract_frames(str(video), tmp_path / "frames", 1, max_frame_size=64)
    assert len(frames) == 4 and all(Path(f).is_file() for f in frames)    # t = 0,1,2,3 s
    assert ortho.extract_frames(str(video), tmp_path / "frames", 1, max_frame_size=64) == frames
    with pytest.raises(RuntimeError, match="settings/input/version changed"):
        ortho.extract_frames(str(video), tmp_path / "frames", 2, max_frame_size=64)
    with pytest.raises(FileNotFoundError):
        ortho.extract_frames(str(tmp_path / "nope.mp4"), tmp_path / "f2")


# ----- build_orthomosaic ----------------------------------------------------------------------
def test_build_raises_unavailable_without_rasterio(tmp_path):
    with patch.dict(sys.modules, {"rasterio": None}):
        with pytest.raises(ortho.OrthomosaicUnavailable, match="rasterio"):
            ortho.build_orthomosaic("v.mp4", tmp_path)


def test_build_raises_unavailable_when_docker_or_image_is_missing(tmp_path):
    with patch.dict(sys.modules, {"rasterio": MagicMock()}), \
            patch("core.azure.orthomosaic.resolve_engine", side_effect=RuntimeError("Docker not found")), \
            patch("core.azure.orthomosaic.extract_frames") as extract:
        with pytest.raises(ortho.OrthomosaicUnavailable, match="Docker not found"):
            ortho.build_orthomosaic("v.mp4", tmp_path)
    extract.assert_not_called()           # nothing heavy runs on a box without the engine


def test_build_returns_a_jpeg_and_report(tmp_path):
    from PIL import Image
    import io

    tif = tmp_path / "project" / "odm_orthophoto" / "odm_orthophoto.tif"
    tif.parent.mkdir(parents=True)
    (tmp_path / "project" / "orthomosaic_report.json").write_text('{"result": {"width": 300}}')
    bgr = np.zeros((40, 60, 3), np.uint8)
    bgr[:, :, 2] = 255                                              # red in BGR
    with patch.dict(sys.modules, {"rasterio": MagicMock()}), \
            patch("core.azure.orthomosaic.resolve_engine", return_value={}), \
            patch("core.azure.orthomosaic.extract_frames", return_value=["f"]) as extract, \
            patch("core.azure.orthomosaic.reconstruct", return_value=tif) as reconstruct, \
            patch("core.azure.orthomosaic.read_preview", return_value=bgr):
        jpeg, report = ortho.build_orthomosaic("v.mp4", tmp_path, long_edge=3000, fast=True)
    assert jpeg[:2] == b"\xff\xd8"
    image = Image.open(io.BytesIO(jpeg))
    assert image.size == (60, 40) and image.getpixel((5, 5))[0] > 200   # converted to RGB (red)
    assert report["output"] == {"width": 60, "height": 40, "bytes": len(jpeg)}
    assert report["result"]["width"] == 300
    assert reconstruct.call_args.kwargs["long_edge"] == 3000 and reconstruct.call_args.kwargs["fast"] is True
    assert extract.call_args.args[1] == (tmp_path.resolve() / "frames")


# ----- background job ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fresh_job_state():
    cache.clear()
    jobs._active.clear()
    yield
    cache.clear()
    jobs._active.clear()


@pytest.fixture
def storage():
    """Patch the blob seams the job uses; yields the mocks."""
    with patch("core.azure.blob.download_blob_to_temp") as download, \
            patch("core.azure.blob.put_blob_and_sas") as put, \
            patch("apps.videos.orthomosaic_jobs.AzureEnvironmentConfig") as cfg:
        def fake_download(url):
            fd, path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            return path

        download.side_effect = fake_download
        put.side_effect = lambda c, name, data, **k: f"https://x/input/{name}?sig=1"
        cfg.from_settings.return_value = SimpleNamespace(input_container="input")
        yield SimpleNamespace(download=download, put=put, cfg=cfg)


def test_run_job_uploads_to_the_frame_folder_and_records_completed(storage, tmp_path):
    engine = {"backend": "docker"}
    with patch("core.azure.orthomosaic.resolve_engine", return_value=engine), \
            patch("core.azure.orthomosaic.build_orthomosaic",
                  return_value=(b"\xff\xd8jpeg", {"output": {"width": 60, "height": 40}})) as build:
        state = jobs.run_job("7", 42, SAS)
    assert state["state"] == jobs.COMPLETED
    assert state["blob"] == "7/images/42/orthomosaic.jpg" and "7/images/42/orthomosaic.jpg" in state["downloadUrl"]
    assert storage.put.call_args.args[1] == "7/images/42/orthomosaic.jpg"
    assert storage.put.call_args.args[2] == b"\xff\xd8jpeg"
    assert build.call_args.kwargs["long_edge"] == 6000                     # script defaults
    assert jobs.get_state("7", 42)["state"] == jobs.COMPLETED
    video, work = build.call_args.args
    assert not Path(video).exists() and not Path(work).exists()            # temp files cleaned up


def test_run_job_without_docker_fails_fast_and_cleanly(storage):
    with patch("core.azure.orthomosaic.resolve_engine", side_effect=RuntimeError("Docker not found.")):
        state = jobs.run_job("7", 42, SAS)          # must not raise
    assert state["state"] == jobs.FAILED and state["unavailable"] is True
    assert "Docker" in state["reason"]
    storage.download.assert_not_called()           # no 150 MB download on a box that cannot run ODM
    storage.put.assert_not_called()
    assert ("7", "42") not in jobs._active


def test_run_job_swallows_any_other_failure(storage):
    with patch("core.azure.orthomosaic.resolve_engine", return_value={}), \
            patch("core.azure.orthomosaic.build_orthomosaic", side_effect=RuntimeError("ODM exited with 137")):
        state = jobs.run_job("7", 42, SAS)
    assert state["state"] == jobs.FAILED and state["unavailable"] is False and "137" in state["reason"]
    storage.put.assert_not_called()


def test_start_job_reuses_completed_and_running_work():
    with patch.object(jobs, "status", return_value={"state": jobs.COMPLETED, "downloadUrl": "u"}), \
            patch.object(jobs.threading, "Thread") as thread:
        assert jobs.start_job("7", 42, SAS)["state"] == jobs.COMPLETED
        thread.assert_not_called()
    jobs._active.add(("7", "42"))
    with patch.object(jobs, "status", return_value={"state": "none"}), \
            patch.object(jobs.threading, "Thread") as thread:
        jobs.start_job("7", 42, SAS)
        thread.assert_not_called()                     # one job per video at a time


def test_start_job_spawns_a_daemon_thread_and_force_regenerates():
    with patch.object(jobs, "status", return_value={"state": jobs.COMPLETED}), \
            patch.object(jobs.threading, "Thread") as thread:
        state = jobs.start_job("7", 42, SAS, force=True)
    assert state["state"] == jobs.QUEUED
    assert thread.call_args.kwargs["daemon"] is True and thread.call_args.kwargs["args"] == ("7", 42, SAS)
    thread.return_value.start.assert_called_once()


def test_start_job_wait_runs_inline_and_without_source_fails():
    with patch.object(jobs, "status", return_value={"state": "none"}), \
            patch.object(jobs, "run_job", return_value={"state": jobs.COMPLETED}) as run:
        assert jobs.start_job("7", 42, SAS, wait=True)["state"] == jobs.COMPLETED
        run.assert_called_once_with("7", 42, SAS)
    assert jobs.start_job("7", 42, None)["state"] == jobs.FAILED


def test_status_resolves_completed_from_the_blob():
    with patch.object(jobs, "AzureEnvironmentConfig"), \
            patch.object(jobs, "result_url", return_value="https://x/o.jpg?sig"):
        state = jobs.status("7", 42)
    assert state == {"state": jobs.COMPLETED, "downloadUrl": "https://x/o.jpg?sig"}
    with patch.object(jobs, "AzureEnvironmentConfig"), patch.object(jobs, "result_url", return_value=None):
        assert jobs.status("7", 42)["state"] == "none"


@override_settings(DVSA_ORTHOMOSAIC_AUTO="false")
def test_auto_flag_off():
    assert jobs.auto_enabled() is False


# ----- endpoint ---------------------------------------------------------------------------------
class OrthomosaicEndpoint(TestCase):
    url = "/api/v1/videos/orthomosaic/"

    def setUp(self):
        cache.clear()
        self.client = APIClient()
        self.user = User.objects.create_user(email="ortho@example.com", password="testpass123",
                                             first_name="Or", last_name="Tho")
        self.client.force_authenticate(user=self.user)

    def _video(self, account="7"):
        with override_settings(DVSA_ORTHOMOSAIC_AUTO="false"):
            entity = VideoEntity()
            entity.create_video(account_id=account, sas_url=SAS)
        return entity

    def test_put_queues_a_background_job_for_the_latest_video(self):
        entity = self._video()
        queued = {"state": jobs.QUEUED}
        with patch.object(jobs, "start_job", return_value=queued) as start:
            resp = self.client.put(self.url, {"account_id": "7"})
        assert resp.status_code == status.HTTP_202_ACCEPTED
        body = resp.json()
        assert body["video_pk"] == entity.pk and body["state"] == jobs.QUEUED
        assert start.call_args.args == ("7", entity.pk, SAS)
        assert start.call_args.kwargs == {"force": False, "wait": False}

    def test_put_returns_the_download_url_when_it_already_exists(self):
        entity = self._video()
        done = {"state": jobs.COMPLETED, "downloadUrl": f"https://x/input/7/images/{entity.pk}/orthomosaic.jpg?sig"}
        with patch.object(jobs, "start_job", return_value=done):
            resp = self.client.put(self.url, {"account_id": "7", "video_id": str(entity.pk)})
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["downloadUrl"].endswith("orthomosaic.jpg?sig")

    def test_put_passes_force_and_wait_flags(self):
        self._video()
        with patch.object(jobs, "start_job", return_value={"state": jobs.COMPLETED}) as start:
            self.client.put(self.url, {"account_id": "7", "force": "true", "wait": "1"})
        assert start.call_args.kwargs == {"force": True, "wait": True}

    def test_missing_docker_is_503_and_other_failures_500(self):
        self._video()
        with patch.object(jobs, "start_job", return_value={
                "state": jobs.FAILED, "reason": "Docker not found.", "unavailable": True}):
            resp = self.client.put(self.url, {"account_id": "7", "wait": "true"})
        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Docker" in resp.json()["reason"]
        with patch.object(jobs, "start_job", return_value={
                "state": jobs.FAILED, "reason": "ODM exited with 137", "unavailable": False}):
            resp = self.client.put(self.url, {"account_id": "7", "wait": "true"})
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_explicit_sas_url_defaults_video_pk_to_zero(self):
        with patch.object(jobs, "start_job", return_value={"state": jobs.QUEUED}) as start:
            resp = self.client.put(self.url, {"account_id": "9", "sas_url": SAS})
        assert resp.status_code == status.HTTP_202_ACCEPTED and resp.json()["video_pk"] == 0
        assert start.call_args.args == ("9", 0, SAS)

    def test_validation_and_auth(self):
        assert self.client.put(self.url, {}).status_code == status.HTTP_400_BAD_REQUEST
        assert self.client.put(self.url, {"account_id": "7", "video_id": "999999"}).status_code == 404
        assert self.client.put(self.url, {"account_id": "none-here"}).status_code == 404
        assert self.client.get(self.url).status_code == status.HTTP_400_BAD_REQUEST
        anon = APIClient()
        assert anon.put(self.url, {"account_id": "7"}).status_code in (401, 403)
        assert anon.get(self.url, {"account_id": "7"}).status_code in (401, 403)

    def test_get_reports_job_state(self):
        entity = self._video()
        with patch.object(jobs, "status", return_value={"state": jobs.RUNNING}) as status_fn:
            resp = self.client.get(self.url, {"account_id": "7"})
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json() == {"account_id": "7", "video_pk": entity.pk, "state": jobs.RUNNING}
        assert status_fn.call_args.args == ("7", entity.pk)
        assert self.client.get(self.url, {"account_id": "nobody"}).status_code == 404


# ----- upload hook --------------------------------------------------------------------------------
class UploadTrigger(TestCase):
    """A new VideoEntity starts the job in the background and can never fail the upload."""

    def _create(self):
        entity = VideoEntity()
        entity.create_video(account_id="7", sas_url=SAS)
        return entity

    @override_settings(DVSA_ORTHOMOSAIC_AUTO="true")
    def test_new_video_starts_the_job(self):
        with patch.object(jobs, "start_job") as start:
            entity = self._create()
        start.assert_called_once_with("7", entity.pk, SAS)

    @override_settings(DVSA_ORTHOMOSAIC_AUTO="true")
    def test_a_failing_job_start_never_breaks_the_upload(self):
        with patch.object(jobs, "start_job", side_effect=RuntimeError("no docker, no cry")):
            entity = self._create()                 # must not raise
        assert VideoEntity.objects.filter(pk=entity.pk).exists()

    @override_settings(DVSA_ORTHOMOSAIC_AUTO="false")
    def test_flag_off_starts_nothing(self):
        with patch.object(jobs, "start_job") as start:
            self._create()
        start.assert_not_called()

    @override_settings(DVSA_ORTHOMOSAIC_AUTO="true")
    def test_updates_and_videos_without_a_url_start_nothing(self):
        with patch.object(jobs, "start_job") as start:
            entity = self._create()
            start.reset_mock()
            entity.save()                           # not a creation
            VideoEntity.objects.create(account_id="7", sas_url="")
        start.assert_not_called()
