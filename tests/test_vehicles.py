"""Tests for vehicle tracking: ``core.vehicles`` and the ``/api/v1/videos/vehicles/`` endpoint.

Fully offline: no ultralytics/torch, model weights or Azure. Pure parts (box helpers, frame
selection, camera motion, tracker, activity, rendering) are tested on synthetic data; the pipeline
runs end to end on a synthetic panning video with a colour detector (needs ffmpeg for the H.264
output test, skipped otherwise); the endpoint tests patch the download, pipeline and upload seams.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from apps.users.models import User
from apps.videos import vehicles as vehicle_service
from apps.videos.models import VideoEntity
from core.vehicles import boxes as box_ops
from core.vehicles.activity import MOVING, STATIONARY, UNKNOWN, ActivityConfig, TrackActivity, estimate_motion
from core.vehicles.detector import VehiclesUnavailable, runtime_missing
from core.vehicles.ffmpeg import find_ffmpeg
from core.vehicles.motion import CameraMotionEstimator, apply_h, transform_boxes
from core.vehicles.pipeline import VehiclePipelineConfig, run_vehicle_pipeline
from core.vehicles.render import PanelInfo, VehicleDraw, draw_frame
from core.vehicles.tracker import TrackerConfig, VehicleTracker
from core.vehicles.video_io import FrameSelector, Mp4Writer, even_size, probe_video
from tests.vehicle_fixtures import ColourDetector, SynthVehicle, make_scene_video, texture

needs_ffmpeg = pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")


def box(cx, cy, w=40.0, h=20.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ----- frame selection / writer ---------------------------------------------------------------
def test_frame_selector_halves_30fps_and_keeps_frame_after_gap():
    ts = [i / 30 for i in range(30)] + [1.5, 1.5 + 1 / 30]
    sel = FrameSelector(15, 30)
    picked = [t for t in ts if sel(t)]
    assert len([t for t in ts[:30] if t in picked]) == 15
    assert 1.5 in picked
    assert all(sel2(t) for sel2 in [FrameSelector(0, 30)] for t in ts)  # 0 = every frame


def test_writer_repeats_frames_to_keep_vfr_timing():
    w = Mp4Writer.__new__(Mp4Writer)
    w.fps, w.n_written = 30.0, 0
    w._emit = lambda f: setattr(w, "n_written", w.n_written + 1)
    ts = [0, 1 / 30, 2 / 30, 2 / 30 + 0.5, 2 / 30 + 0.5 + 1 / 30]
    for t_next in ts[1:] + [ts[-1] + 1 / 30]:
        w.write(None, t_next)
    assert w.n_written == round((ts[-1] + 1 / 30) * 30)


def test_even_size_rounds_down():
    assert even_size(1281, 721) == (1280, 720)


# ----- motion ---------------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["similarity", "homography"])
def test_motion_estimator_recovers_known_transform(model):
    base = texture(360, 640)
    M = cv2.getRotationMatrix2D((320, 180), 1.5, 1.03)
    M[:, 2] += (6.0, -4.0)
    nxt = cv2.warpAffine(base, M, (640, 360), borderMode=cv2.BORDER_REFLECT)
    est = CameraMotionEstimator(model)
    est.update(base)
    step = est.update(nxt)
    assert step.ok
    pts = np.array([[100, 100], [500, 80], [320, 300]], float)
    M3 = np.vstack([M, [0, 0, 1]])
    assert np.abs(apply_h(step.H_prev_to_cur, pts) - apply_h(M3, pts)).max() < 0.3
    assert np.abs(apply_h(step.H_cur_to_ref, apply_h(M3, pts)) - pts).max() < 0.3


def test_motion_ignores_masked_moving_object():
    base = texture(360, 640, seed=1)
    a, b = base.copy(), np.roll(base, 5, axis=1)  # background shifts +5 px in x
    cv2.rectangle(a, (200, 100), (440, 260), (0, 0, 230), -1)
    cv2.rectangle(b, (200, 140), (440, 300), (0, 0, 230), -1)  # big object moves differently
    est = CameraMotionEstimator("similarity")
    est.update(a, np.array([[200, 100, 440, 260]]))
    step = est.update(b, np.array([[200, 140, 440, 300]]))
    assert step.ok
    assert abs(step.H_prev_to_cur[0, 2] - 5.0) < 0.3 and abs(step.H_prev_to_cur[1, 2]) < 0.3


def test_failed_registration_reuses_last_motion():
    base = texture(360, 640, seed=2)
    est = CameraMotionEstimator("similarity")
    est.update(base)
    good = est.update(np.roll(base, 4, axis=1))
    flat = np.full_like(base, 128)  # no features at all
    bad = est.update(flat)
    assert good.ok and not bad.ok
    assert np.allclose(bad.H_prev_to_cur, good.H_prev_to_cur)


def test_transform_boxes_identity_and_shift():
    b = np.array([box(50, 40)])
    assert np.allclose(transform_boxes(np.eye(3), b), b)
    T = np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1]], float)
    assert np.allclose(transform_boxes(T, b), b + [5, -3, 5, -3])


# ----- box helpers ----------------------------------------------------------------------------
def test_tiles_cover_image_and_edge_boxes_are_dropped():
    wins = box_ops.tile_windows(3000, 1700, 1280, 0.2)
    cover = np.zeros((1700, 3000), bool)
    for x0, y0, x1, y1 in wins:
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    m = box_ops.touches_interior_edge(np.array([[1270, 10, 1290, 30], [100, 100, 140, 120]], float),
                                      (0, 0, 1280, 1280), 3000, 1700)
    assert m.tolist() == [True, False]


def test_nms_keeps_best_of_overlaps():
    b = np.array([box(50, 50), box(52, 50), box(200, 50)], float)
    keep = box_ops.nms(b, np.array([0.5, 0.9, 0.7]), 0.5)
    assert keep.tolist() == [1, 2]


# ----- tracker --------------------------------------------------------------------------------
def run_tracker(frames, cfg=None, dt=1 / 15):
    """frames: list of list of (cx, cy, score[, class]). Returns per-frame {det_index: id} and tracker."""
    tr = VehicleTracker(cfg or TrackerConfig())
    out = []
    for k, dets in enumerate(frames):
        b = np.array([box(d[0], d[1]) for d in dets], float).reshape(-1, 4)
        s = np.array([d[2] for d in dets], float)
        c = [d[3] if len(d) > 3 else "car" for d in dets]
        out.append({j: t.id for t, j in tr.update(k * dt, k, b, b, s, c)})
    return out, tr


def test_crossing_vehicles_keep_their_ids():
    frames = [[(100 + 6 * k, 100, 0.9), (300 - 6 * k, 104, 0.9)] for k in range(40)]
    out, _ = run_tracker(frames)
    assert out[5] == {0: 1, 1: 2}
    assert out[-1] == {0: 1, 1: 2}  # after crossing, no swap


def test_single_frame_false_positive_gets_no_id():
    frames = [[(100, 100, 0.9)] for _ in range(10)]
    frames[4] = frames[4] + [(400, 300, 0.9)]
    out, tr = run_tracker(frames)
    assert tr.n_confirmed == 1
    assert all(set(o.values()) <= {1} for o in out)


def test_short_gap_is_refound_long_gap_gets_new_id():
    moving = lambda k: (100 + 3 * k, 100, 0.9)  # noqa: E731
    frames = [[moving(k)] for k in range(60)]
    for k in range(20, 27):  # ~0.47 s gap
        frames[k] = []
    out, _ = run_tracker(frames)
    assert out[30] == {0: 1}
    frames = [[moving(k)] for k in range(80)]
    for k in range(20, 60):  # ~2.7 s gap > max_lost_s
        frames[k] = []
    out, _ = run_tracker(frames)
    assert out[70] == {0: 2}


def test_low_score_detections_continue_but_never_start_tracks():
    frames = [[(100, 100, 0.9)] for _ in range(5)] + [[(100, 100, 0.15), (400, 300, 0.15)] for _ in range(10)]
    out, tr = run_tracker(frames)
    assert out[10] == {0: 1}
    assert tr.n_confirmed == 1


def test_track_class_is_majority_vote():
    frames = [[(100, 100, 0.9, "car")] for _ in range(4)] + [[(100, 100, 0.9, "truck")]]
    _, tr = run_tracker(frames)
    assert tr.finalize()[0].class_name == "car"


# ----- activity -------------------------------------------------------------------------------
def test_activity_states():
    cfg = ActivityConfig()
    rng = np.random.default_rng(0)
    ts = np.arange(0, 1.0, 1 / 15)
    still = np.c_[100 + rng.normal(0, 0.5, len(ts)), 100 + rng.normal(0, 0.5, len(ts))]
    assert TrackActivity(cfg).update(ts[-1], ts, still, np.full(len(ts), 40.0), True) == STATIONARY
    moving = np.c_[100 + 40 * ts, 100 + 0 * ts]  # 1 BL/s to the right
    act = TrackActivity(cfg)
    assert act.update(ts[-1], ts, moving, np.full(len(ts), 40.0), True) == MOVING
    assert abs(act.estimate.heading_deg) < 1.0
    assert TrackActivity(cfg).update(ts[2], ts[:3], moving[:3], np.full(3, 40.0), True) == UNKNOWN
    assert TrackActivity(cfg).update(ts[-1], ts, moving, np.full(len(ts), 40.0), False) == UNKNOWN


def test_activity_switch_needs_min_dwell():
    cfg = ActivityConfig(min_dwell_s=0.5)
    act = TrackActivity(cfg)
    t_all = np.arange(0, 4, 1 / 15)
    x = np.where(t_all < 2.0, 100 + 40 * t_all, 180.0)  # drives 2 s, then stops
    states = []
    for i in range(len(t_all)):
        sel = slice(0, i + 1)
        states.append(act.update(t_all[i], t_all[sel], np.c_[x[sel], np.full(i + 1, 50.0)],
                                 np.full(i + 1, 40.0), True))
    stop_idx = next(i for i, s in enumerate(states) if i > 15 and s == STATIONARY)
    assert t_all[stop_idx] >= 2.0 + 0.5  # not before the window clears plus the dwell
    assert states[-1] == STATIONARY and MOVING in states


def test_estimate_motion_speed_in_body_lengths():
    t = np.linspace(0, 1, 16)
    est = estimate_motion(t, np.c_[80 * t, 60 * t], 50.0)
    assert abs(est.speed_bl_s - 2.0) < 1e-6


# ----- render ---------------------------------------------------------------------------------
def test_draw_frame_returns_new_image_with_dark_panel():
    frame = np.full((360, 640, 3), 200, np.uint8)
    item = VehicleDraw(3, "car", np.array(box(300, 200)), MOVING, np.array([[280, 200], [300, 200]]))
    out = draw_frame(frame, [item],
                     PanelInfo(1.0, 4.0, 15, 1, 1, 3, True, {MOVING: 1}, {"car": 1}), (640, 360))
    assert out.shape == frame.shape and frame.mean() == 200
    assert out[20:40, 20:60].mean() < 120  # darkened panel
    small = draw_frame(frame, [], None, (320, 180))
    assert small.shape == (180, 320, 3)


# ----- detector runtime -----------------------------------------------------------------------
def test_auto_imgsz_upscales_frame_long_side():
    from core.vehicles.detector import auto_imgsz

    assert auto_imgsz((720, 1280, 3)) == 1920
    assert auto_imgsz((1080, 1920, 3)) == 2880
    assert auto_imgsz((2160, 3840, 3)) == 3840  # clamped
    assert auto_imgsz(None) == 1920


def test_ultralytics_detector_unavailable_without_runtime():
    from core.vehicles import detector as det

    with patch.object(det, "runtime_missing", return_value="ultralytics"):
        with pytest.raises(VehiclesUnavailable):
            det.UltralyticsVehicleDetector()


# ----- pipeline end to end --------------------------------------------------------------------
@pytest.fixture
def scene_video(tmp_path):
    path = tmp_path / "scene.mp4"
    vehicles = [SynthVehicle(320, 200), SynthVehicle(150, 120, vx=60.0)]  # parked, driving at 1.5 BL/s
    make_scene_video(str(path), vehicles, seconds=4.0)
    return str(path)


def _run(scene_video, out_dir, **kw):
    cfg = VehiclePipelineConfig(**kw)
    return run_vehicle_pipeline(scene_video, str(out_dir), cfg, lambda frame: ColourDetector())


@needs_ffmpeg
def test_pipeline_compensates_camera_motion(scene_video, tmp_path):
    rep = _run(scene_video, tmp_path / "a")
    assert rep["vehicles"]["unique_track_ids"] == 2
    parked = min(rep["tracks"], key=lambda t: t["median_speed_bl_s"])
    driving = max(rep["tracks"], key=lambda t: t["median_speed_bl_s"])
    assert parked["dominant_state"] == "STATIONARY" and parked["median_speed_bl_s"] < 0.2
    assert driving["dominant_state"] == "MOVING"
    assert 1.2 < driving["median_speed_bl_s"] < 1.8  # true speed 60 px/s / 40 px = 1.5 BL/s
    assert len(parked["trajectory_ref_px"]) >= 15  # ~5 Hz over ~4 s

    # Without compensation the drone's pan (40 px/s = 1 BL/s) makes the parked car "move".
    rep_none = _run(scene_video, tmp_path / "b", motion_model="none")
    assert [t["dominant_state"] for t in rep_none["tracks"]] == ["MOVING", "MOVING"]


@needs_ffmpeg
def test_pipeline_writes_browser_playable_mp4_and_report(scene_video, tmp_path):
    rep = _run(scene_video, tmp_path / "out")
    out = tmp_path / "out" / "vehicles_tracked.mp4"
    info = probe_video(str(out))
    src = probe_video(scene_video)
    assert (info.width, info.height) == (640, 360)
    assert abs(info.duration_s - src.duration_s) <= 1.5 / src.fps_nominal
    assert rep["output"]["codec"] == "h264" and rep["output"]["browser_playable"]
    saved = json.loads((tmp_path / "out" / "vehicles_tracking.json").read_text())
    assert saved["vehicles"]["by_class"]["car"] == 2
    assert rep["video"]["frames_processed"] == 60  # 4 s at 15 fps


@needs_ffmpeg
def test_pipeline_time_range_and_empty_range(scene_video, tmp_path):
    rep = _run(scene_video, tmp_path / "r", start_s=1.0, end_s=2.0)
    assert 0.9 < rep["video"]["processed_range_s"][1] - rep["video"]["processed_range_s"][0] <= 1.0
    with pytest.raises(ValueError):
        _run(scene_video, tmp_path / "e", start_s=10.0)


# ----- endpoint -------------------------------------------------------------------------------
def _fake_sas(cfg, blob_name, data, **kwargs):
    return f"https://sadronevideo.blob.core.windows.net/input/{blob_name}?sig=x"


def _fake_pipeline(video_path, out_dir, cfg, make_detector, progress=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vehicles_tracked.mp4"), "wb") as fh:
        fh.write(b"\x00\x00\x00\x18ftypmp42")
    rep = {"video": {"duration_s": 4.0}, "vehicles": {"unique_track_ids": 2}, "output": {"codec": "h264"},
           "timing_s": {"total": 1.0}, "performance": {}, "tracks": []}
    with open(os.path.join(out_dir, "vehicles_tracking.json"), "w") as fh:
        json.dump(rep, fh)
    _fake_pipeline.cfg = cfg
    return rep


class VehiclesEndpoint(TestCase):
    url = "/api/v1/videos/vehicles/"

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(email="vehicles@example.com", password="testpass123",
                                             first_name="Veh", last_name="Icles")
        self.entity = VideoEntity()
        self.entity.create_video(account_id="5", sas_url="https://sadronevideo.blob.core.windows.net/input/5/v.mp4")
        self.tmp_downloads = []

    def _download(self, url):
        path = os.path.join(os.environ.get("TEMP", "/tmp"), f"vehicles-test-{len(self.tmp_downloads)}.mp4")
        with open(path, "wb") as fh:
            fh.write(b"x")
        self.tmp_downloads.append(path)
        return path

    def _put(self, data, *, existing=None, missing=None, pipeline=_fake_pipeline):
        self.client.force_authenticate(user=self.user)
        with patch("apps.videos.vehicles.existing_result", return_value=existing) as ex, \
                patch("core.vehicles.detector.runtime_missing", return_value=missing), \
                patch("core.azure.blob.download_blob_to_temp", side_effect=self._download) as dl, \
                patch("core.vehicles.pipeline.run_vehicle_pipeline", side_effect=pipeline) as run, \
                patch("core.azure.blob.put_blob_and_sas", side_effect=_fake_sas) as sas:
            resp = self.client.put(self.url, data, format="json")
        return resp, {"existing": ex, "download": dl, "run": run, "sas": sas}

    def test_requires_authentication(self):
        resp = self.client.put(self.url, {"account_id": "5", "video_id": self.entity.pk}, format="json")
        assert resp.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_validation_errors(self):
        self.client.force_authenticate(user=self.user)
        assert self.client.put(self.url, {"video_id": 1}, format="json").status_code == 400
        assert self.client.put(self.url, {"account_id": "5"}, format="json").status_code == 400
        bad = {"account_id": "5", "video_id": self.entity.pk}
        assert self.client.put(self.url, {**bad, "start_s": "abc"}, format="json").status_code == 400
        assert self.client.put(self.url, {**bad, "start_s": 5, "end_s": 2}, format="json").status_code == 400
        other = {"account_id": "6", "video_id": self.entity.pk}  # someone else's video
        assert self.client.put(self.url, other, format="json").status_code == 404

    def test_generates_uploads_and_returns_sas_url(self):
        resp, m = self._put({"account_id": "5", "video_id": self.entity.pk})
        assert resp.status_code == status.HTTP_200_OK, resp.content
        body = resp.json()
        blob = f"5/images/{self.entity.pk}/vehicles/vehicles_tracked.mp4"
        assert body["state"] == "completed" and body["cached"] is False
        assert body["downloadUrl"].endswith(f"{blob}?sig=x")
        assert body["reportUrl"].endswith(f"5/images/{self.entity.pk}/vehicles/vehicles_tracking.json?sig=x")
        assert body["summary"]["vehicles"]["unique_track_ids"] == 2
        m["download"].assert_called_once_with(self.entity.sas_url)
        first = m["sas"].call_args_list[0]
        assert first.args[1] == blob and first.kwargs["content_type"] == "video/mp4"
        assert all(not os.path.exists(p) for p in self.tmp_downloads)  # temp video removed

    def test_existing_result_is_reused_unless_forced(self):
        existing = {"state": "completed", "downloadUrl": "https://x/y.mp4?sig=1"}
        resp, m = self._put({"account_id": "5", "video_id": self.entity.pk}, existing=existing)
        assert resp.status_code == 200 and resp.json()["cached"] is True
        m["run"].assert_not_called()
        resp, m = self._put({"account_id": "5", "video_id": self.entity.pk, "force": True}, existing=existing)
        assert resp.status_code == 200 and resp.json()["cached"] is False
        m["run"].assert_called_once()

    def test_time_range_uses_range_blob_and_reaches_config(self):
        resp, m = self._put({"account_id": "5", "video_id": self.entity.pk, "start_s": 1.5, "end_s": 20})
        assert resp.status_code == 200
        assert m["sas"].call_args_list[0].args[1].endswith("vehicles/vehicles_tracked_1p5-20.mp4")
        assert _fake_pipeline.cfg.start_s == 1.5 and _fake_pipeline.cfg.end_s == 20.0

    def test_missing_runtime_returns_503_without_download(self):
        resp, m = self._put({"account_id": "5", "video_id": self.entity.pk}, missing="ultralytics")
        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert resp.json()["unavailable"] is True
        m["download"].assert_not_called()

    def test_pipeline_failure_returns_500_and_cleans_up(self):
        def boom(*a, **k):
            raise RuntimeError("ffmpeg exited with 1")

        resp, _ = self._put({"account_id": "5", "video_id": self.entity.pk}, pipeline=boom)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert resp.json()["unavailable"] is False and "ffmpeg" in resp.json()["reason"]
        assert all(not os.path.exists(p) for p in self.tmp_downloads)

    def test_busy_returns_409(self):
        slots = vehicle_service._semaphore()
        assert slots.acquire(blocking=False)
        try:
            resp, m = self._put({"account_id": "5", "video_id": self.entity.pk})
        finally:
            slots.release()
        assert resp.status_code == status.HTTP_409_CONFLICT
        m["run"].assert_not_called()

    def test_get_reports_none_or_completed(self):
        self.client.force_authenticate(user=self.user)
        q = f"{self.url}?account_id=5&video_id={self.entity.pk}"
        with patch("apps.videos.vehicles.existing_result", return_value=None):
            assert self.client.get(q).json()["state"] == "none"
        with patch("apps.videos.vehicles.existing_result",
                   return_value={"state": "completed", "downloadUrl": "https://x/v.mp4?sig=1"}):
            body = self.client.get(q).json()
        assert body["state"] == "completed" and body["downloadUrl"].startswith("https://")


def test_runtime_missing_reports_a_package_name_or_none():
    assert runtime_missing() in (None, "torch", "ultralytics")


def test_fresh_source_url_resigns_own_account_blobs_only():
    class Cfg:
        storage_account, account_key = "sadronevideo", "a2V5"

    def fake_sas(cfg, blob_name, container=None, ttl_hours=1):
        return f"https://sadronevideo.blob.core.windows.net/{container}/{blob_name}?sig=fresh"

    stale = "https://sadronevideo.blob.core.windows.net/input/5/abc.mp4?se=2026-09-18&sig=old"
    with patch("core.azure.blob.read_sas_for_blob", side_effect=fake_sas):
        fresh = "https://sadronevideo.blob.core.windows.net/input/5/abc.mp4?sig=fresh"
        assert vehicle_service.fresh_source_url(Cfg(), stale) == fresh
        other = "https://elsewhere.blob.core.windows.net/input/5/abc.mp4?sig=x"
        assert vehicle_service.fresh_source_url(Cfg(), other) == other
        Cfg.account_key = None
        assert vehicle_service.fresh_source_url(Cfg(), stale) == stale


def test_blob_names_layout():
    assert vehicle_service.blob_names("5", 2) == {
        "video": "5/images/2/vehicles/vehicles_tracked.mp4",
        "report": "5/images/2/vehicles/vehicles_tracking.json"}
    assert vehicle_service.blob_names("5", 2, 0, None)["video"].endswith("vehicles_tracked_0-end.mp4")
