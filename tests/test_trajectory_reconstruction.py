"""Tests for the ``trajectory_reconstruction`` routine and its SORT tracker.

Fully offline and Django-free: the tracker and routine operate on synthetic
detection sequences (no video runtime, OpenCV, or Azure needed), mirroring the
style of the other routine tests.
"""

from __future__ import annotations

import numpy as np

from apps.analytics.routines import available_routines, get_routine
from apps.analytics.routines.base import Detection
from apps.analytics.routines.tracking import (
    SortTracker,
    iou,
    trajectory_reconstruction_routine,
)


# ----- helpers -------------------------------------------------------------
# Boxes are 30px and objects step 10px/frame, so consecutive frames overlap well
# above the IoU gate (~0.5) — mimicking real dense sampling, where SORT tracks.
def _det(x, y, w=30, h=30, label="object", score=1.0) -> Detection:
    return Detection(bbox=(x, y, w, h), centroid=(x + w / 2, y + h / 2),
                     area=w * h, label=label, score=score)


def _d(x, y, w=30, h=30, label="object", score=1.0) -> dict:
    return {"bbox": [x, y, w, h], "label": label, "score": score}


def _run(dets_per_frame, **kw) -> dict:
    frames = [None] * len(dets_per_frame)
    return trajectory_reconstruction_routine(
        iter(frames), detections_per_frame=dets_per_frame, **kw)


def _reported_ids_per_frame(tracker, dets_per_frame):
    out = []
    for dets in dets_per_frame:
        reported = tracker.update([_det(*d) if isinstance(d, tuple) else d for d in dets])
        out.append([r[0] for r in reported])
    return out


# ----- geometry ------------------------------------------------------------
def test_iou_basic():
    a = np.array([0, 0, 10, 10], dtype=float)
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30], dtype=float)) == 0.0
    # Half-overlap in x, full in y -> 1/3.
    assert abs(iou(a, np.array([5, 0, 15, 10], dtype=float)) - (50 / 150)) < 1e-9


# ----- tracker -------------------------------------------------------------
def test_single_linear_track_keeps_one_id():
    seq = [[_det(x, 50)] for x in range(0, 70, 10)]  # 7 frames moving right
    tracker = SortTracker(min_hits=1)
    ids = _reported_ids_per_frame(tracker, seq)
    flat = [i for frame in ids for i in frame]
    assert flat, "track should be reported"
    assert len(set(flat)) == 1


def test_occlusion_within_max_age_reassociates_same_id():
    seq = ([[_det(x, 50)] for x in (0, 10, 20)]
           + [[], []]                          # 2-frame occlusion
           + [[_det(x, 50)] for x in (50, 60)])
    tracker = SortTracker(min_hits=1, max_age=5)
    ids = _reported_ids_per_frame(tracker, seq)
    flat = [i for frame in ids for i in frame]
    assert len(set(flat)) == 1  # same object throughout


def test_occlusion_past_max_age_gets_new_id():
    seq = ([[_det(x, 50)] for x in (0, 10, 20)]
           + [[], []]                          # gap longer than max_age
           + [[_det(x, 50)] for x in (50, 60)])
    tracker = SortTracker(min_hits=1, max_age=1)
    ids = _reported_ids_per_frame(tracker, seq)
    early = {i for i in ids[0] + ids[1] + ids[2]}
    late = {i for frame in ids[5:] for i in frame}
    assert early and late
    assert early.isdisjoint(late)  # reappearance is a new track


def test_min_hits_suppresses_a_late_one_frame_blip():
    # A real track present every frame, plus a spurious detection at frame 5.
    seq = [[_det(x, 50)] for x in range(0, 60, 10)]  # 6 frames
    seq[5] = [_det(50, 50), _det(500, 500)]          # blip far away, late frame
    out = _run([[_d(*t.bbox) for t in frame] for frame in seq],
               min_hits=3, min_track_len=1)
    # The blip's coordinate must never appear in any trajectory point.
    for tr in out["trajectories"]:
        for p in tr["points"]:
            assert not (p["centroid"][0] > 400 and p["centroid"][1] > 400)


# ----- routine -------------------------------------------------------------
def test_two_objects_produce_two_non_swapped_trajectories():
    # A moves right (y=40), B moves left (y=90); no vertical overlap -> no swap.
    dets = []
    xs_a = list(range(0, 70, 10))  # 0..60
    for xa in xs_a:
        xb = 60 - xa
        dets.append([_d(xa, 40), _d(xb, 90)])
    out = _run(dets, min_hits=1)
    assert out["summary"]["num_tracks"] == 2
    trajs = out["trajectories"]
    left_starter = min(trajs, key=lambda t: t["points"][0]["centroid"][0])
    # The object that started on the left ends on the right (identities not swapped).
    assert left_starter["points"][-1]["centroid"][0] > left_starter["points"][0]["centroid"][0]


def test_stationary_vs_moving_flags():
    dets = []
    for x in range(0, 70, 10):
        dets.append([_d(200, 200), _d(x, 50)])  # first still, second moving
    out = _run(dets, min_hits=1, fps=30)
    by_stationary = {t["is_stationary"]: t for t in out["trajectories"]}
    assert set(by_stationary) == {True, False}
    assert by_stationary[False]["displacement_px"] > by_stationary[True]["displacement_px"]
    # With fps supplied, moving track reports a duration and speed.
    assert by_stationary[False]["duration_s"] is not None
    assert by_stationary[False]["mean_speed_px_s"] is not None


def test_min_track_len_filters_short_tracks():
    dets = [[_d(x, 50)] for x in range(0, 60, 10)]
    dets[3].append(_d(400, 400))  # appears once -> too short
    out = _run(dets, min_hits=1, min_track_len=3)
    assert all(t["num_points"] >= 3 for t in out["trajectories"])


def test_deterministic():
    dets = [[_d(x, 50), _d(90 - x, 90)] for x in range(0, 70, 10)]
    a = _run(dets, min_hits=1)
    b = _run(dets, min_hits=1)
    assert a == b


def test_registered_as_video_routine():
    names = {r["name"]: r for r in available_routines()}
    assert "trajectory_reconstruction" in names
    assert names["trajectory_reconstruction"]["level"] == "video"
    spec = get_routine("trajectory_reconstruction")
    out = spec.func(iter([None, None, None]),
                    detections_per_frame=[[_d(0, 0)], [_d(10, 0)], [_d(20, 0)]],
                    min_hits=1)
    assert out["routine"] == "trajectory_reconstruction"
    assert out["summary"]["num_frames"] == 3
    assert len(out["trajectories"]) == 1


def test_self_contained_color_detection():
    import pytest

    pytest.importorskip("cv2")
    # A red square drifting right on a black background across 5 frames.
    frames = []
    for k in range(5):
        img = np.zeros((80, 160, 3), dtype=np.uint8)
        x = 10 + k * 12
        img[30:55, x:x + 25] = (0, 0, 255)  # BGR red
        frames.append(img)
    out = trajectory_reconstruction_routine(iter(frames), min_hits=1, min_area=50.0)
    assert out["summary"]["num_tracks"] >= 1
