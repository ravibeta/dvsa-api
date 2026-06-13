"""Unit tests for the classical CV/ML vision routines.

These tests build small synthetic frames and point sets so they run quickly and
deterministically without needing real drone footage. They exercise the routine
logic directly (no Django/Celery required).
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from apps.analytics.routines import available_routines, get_routine, run_frame_routine
from apps.analytics.routines.clustering import cluster_points
from apps.analytics.routines.detection import detect_by_color
from apps.analytics.routines.geometry import estimate_homography, map_points
from apps.analytics.routines.histograms import rgb_histogram
from apps.analytics.routines.motion import background_subtraction, dense_optical_flow
from apps.analytics.routines.parking import classify_spots, spot_features
from apps.analytics.routines.tiling import non_max_suppression, slice_frame, sliced_detection
from apps.analytics.routines.zone_counting import CentroidTracker, count_in_zones
from apps.analytics.routines.base import Detection


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _frame_with_red_squares(n=3, size=400, square=40):
    """Black frame with ``n`` solid red squares (BGR)."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    centres = []
    for i in range(n):
        x = 30 + i * 100
        y = 50
        frame[y:y + square, x:x + square] = (0, 0, 255)  # red in BGR
        centres.append((x + square / 2, y + square / 2))
    return frame, centres


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_registry_lists_expected_routines():
    names = {r["name"] for r in available_routines()}
    expected = {
        "color_detection", "threshold_detection", "zone_counting",
        "parking_occupancy", "spatial_clustering", "color_histogram",
        "optical_flow", "background_subtraction", "homography_map",
        "sliced_detection",
    }
    assert expected.issubset(names)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def test_detect_by_color_finds_all_squares():
    frame, centres = _frame_with_red_squares(n=3)
    dets = detect_by_color(frame, [0, 70, 50], [10, 255, 255], min_area=50)
    assert len(dets) == 3
    # Centroids should be close to the true square centres.
    found = sorted(d.centroid[0] for d in dets)
    expected = sorted(c[0] for c in centres)
    for f, e in zip(found, expected):
        assert abs(f - e) < 5


def test_color_detection_routine_envelope():
    frame, _ = _frame_with_red_squares(n=2)
    out = run_frame_routine("color_detection", frame, min_area=50)
    assert out["routine"] == "color_detection"
    assert out["summary"]["count"] == 2
    assert len(out["detections"]) == 2


# --------------------------------------------------------------------------- #
# Zone counting + tracker
# --------------------------------------------------------------------------- #

def test_count_in_zones():
    dets = [
        Detection(bbox=(0, 0, 10, 10), centroid=(5, 5), area=100),
        Detection(bbox=(0, 0, 10, 10), centroid=(150, 150), area=100),
    ]
    zones = [
        {"name": "left", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
        {"name": "right", "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]},
    ]
    counts = count_in_zones(dets, zones)
    assert counts == {"left": 1, "right": 1}


def test_centroid_tracker_keeps_ids_stable():
    tracker = CentroidTracker(max_distance=30)
    objs = tracker.update([(10, 10), (100, 100)])
    ids_first = set(objs.keys())
    # Move each centroid slightly; IDs must be preserved.
    objs2 = tracker.update([(12, 11), (98, 102)])
    assert set(objs2.keys()) == ids_first


# --------------------------------------------------------------------------- #
# Parking occupancy
# --------------------------------------------------------------------------- #

def test_parking_heuristic_distinguishes_occupied_from_empty():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    # "occupied" spot: noisy texture -> high edge density.
    rng = np.random.default_rng(0)
    frame[0:100, 0:100] = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    # "empty" spot: flat gray -> low edge density.
    frame[0:100, 200:300] = (120, 120, 120)

    occupied_feat = spot_features(frame, (0, 0, 100, 100))
    empty_feat = spot_features(frame, (200, 0, 100, 100))
    assert occupied_feat[0] > empty_feat[0]

    results = classify_spots(
        frame, [(0, 0, 100, 100), (200, 0, 100, 100)], edge_threshold=0.05
    )
    assert results[0]["occupied"] is True
    assert results[1]["occupied"] is False


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

def test_cluster_points_two_groups():
    rng = np.random.default_rng(1)
    g1 = rng.normal(loc=(10, 10), scale=1.0, size=(20, 2))
    g2 = rng.normal(loc=(100, 100), scale=1.0, size=(20, 2))
    pts = np.vstack([g1, g2])
    res = cluster_points(pts, algorithm="dbscan", eps=5, min_samples=3)
    assert res["n_clusters"] == 2


def test_cluster_points_empty():
    res = cluster_points([], algorithm="dbscan")
    assert res["n_clusters"] == 0


# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #

def test_rgb_histogram_normalised():
    frame, _ = _frame_with_red_squares(n=1)
    hist = rgb_histogram(frame, bins=16)
    assert set(hist["histogram"].keys()) == {"b", "g", "r"}
    for ch in ("b", "g", "r"):
        assert pytest.approx(sum(hist["histogram"][ch]), abs=1e-6) == 1.0


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #

def test_background_subtraction_detects_moving_object():
    frames = []
    for i in range(15):
        f = np.zeros((200, 200, 3), dtype=np.uint8)
        x = 10 + i * 10  # a white box sliding across the frame
        f[80:120, x:x + 30] = 255
        frames.append(f)
    res = background_subtraction(frames, min_area=50)
    assert res["frames_processed"] == 15
    # At least some later frame should detect the moving box.
    assert any(fr["moving_objects"] for fr in res["per_frame"])


def test_optical_flow_detects_shift():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    a[40:60, 40:60] = 255
    b = np.zeros((100, 100, 3), dtype=np.uint8)
    b[40:60, 50:70] = 255  # shifted right by 10px
    flow = dense_optical_flow(a, b)
    assert flow["mean_magnitude"] > 0


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def test_homography_identity_like_mapping():
    src = [[0, 0], [100, 0], [100, 100], [0, 100]]
    dst = [[0, 0], [200, 0], [200, 200], [0, 200]]  # 2x scale
    H, mask = estimate_homography(src, dst)
    mapped = map_points(H, [[50, 50]])
    assert mapped[0][0] == pytest.approx(100, abs=1e-3)
    assert mapped[0][1] == pytest.approx(100, abs=1e-3)


def test_homography_requires_four_points():
    with pytest.raises(ValueError):
        estimate_homography([[0, 0], [1, 1]], [[0, 0], [1, 1]])


# --------------------------------------------------------------------------- #
# Tiling / SAHI
# --------------------------------------------------------------------------- #

def test_slice_frame_covers_image():
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    tiles = slice_frame(frame, tile_size=(400, 400), overlap=0.2)
    assert len(tiles) > 1
    # Every tile origin must be within the frame.
    for t in tiles:
        assert 0 <= t["x"] <= 1000
        assert 0 <= t["y"] <= 1000


def test_nms_removes_duplicates():
    a = Detection(bbox=(0, 0, 100, 100), centroid=(50, 50), area=10000, score=0.9)
    b = Detection(bbox=(5, 5, 100, 100), centroid=(55, 55), area=10000, score=0.5)
    kept = non_max_suppression([a, b], iou_threshold=0.4)
    assert len(kept) == 1
    assert kept[0].score == 0.9


def test_sliced_detection_merges_to_full_frame():
    # Large frame with red squares; tiled detection should recover them.
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for (cx, cy) in [(100, 100), (800, 800)]:
        frame[cy:cy + 30, cx:cx + 30] = (0, 0, 255)

    def detector(tile):
        return detect_by_color(tile, [0, 70, 50], [10, 255, 255], min_area=20)

    dets = sliced_detection(frame, detector, tile_size=(400, 400), overlap=0.2)
    assert len(dets) == 2
