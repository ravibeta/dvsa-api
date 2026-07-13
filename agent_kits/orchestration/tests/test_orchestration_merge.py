"""Orchestration tests — partitioning, deterministic merge, and critic (offline)."""

import json
import os

import pytest

from agent_kits.common import Detection, RunInput, RunOutput
from agent_kits.orchestration.child_worker_template import process_partition
from agent_kits.orchestration.critic_review import (
    review,
    sample_for_human_review,
    secondary_model_validation,
)
from agent_kits.orchestration.merge_outputs import canonical_timeline, merge_run_outputs
from agent_kits.orchestration.parent_orchestrator import (
    ParentOrchestrator,
    partition_time_windows,
    partition_tiles,
)
from agent_kits.test_assets.generate_synthetic_video import ensure_manifest

_ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "test_assets", "sample_short.json")


@pytest.fixture(scope="module")
def video() -> str:
    return ensure_manifest(_ASSET)


def _ri(video: str) -> RunInput:
    return RunInput(video_uri=video, processing_flags={"fps": 5}, run_id="base")


def _det(ts, box, cls="car", conf=0.9) -> Detection:
    return Detection(timestamp=ts, bbox=box, class_name=cls, confidence=conf)


# --------------------------------------------------------------------------- #
# Partitioning
# --------------------------------------------------------------------------- #
def test_time_windows_cover_and_overlap():
    windows = partition_time_windows(0.0, 10.0, 5, overlap=0.2)
    assert len(windows) == 5
    assert float(windows[0].start) == 0.0
    assert float(windows[-1].end) == 10.0
    # Overlap: window 1 starts before window 0 ends.
    assert float(windows[1].start) < float(windows[0].end)


def test_tiles_grid_and_overlap():
    tiles = partition_tiles(640, 480, 2, 2, overlap=0.1)
    assert len(tiles) == 4
    assert all({"x", "y", "w", "h"} <= set(t) for t in tiles)
    # Tiles overlap → summed area exceeds the frame area.
    assert sum(t["w"] * t["h"] for t in tiles) > 640 * 480


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        partition_time_windows(0, 10, 3, overlap=1.5)


# --------------------------------------------------------------------------- #
# Merge determinism + conflict resolution
# --------------------------------------------------------------------------- #
def test_merge_removes_iou_overlaps_keeping_highest_confidence():
    a = RunOutput(run_id="a", start_time="s", end_time="e",
                  detections=[_det(1.0, [0, 0, 10, 10], conf=0.6)],
                  summary={"frames_processed": 2})
    b = RunOutput(run_id="b", start_time="s", end_time="e",
                  detections=[_det(1.0, [1, 1, 10, 10], conf=0.9)],  # overlaps a
                  summary={"frames_processed": 2})
    merged = merge_run_outputs([a, b], iou_threshold=0.5)
    assert len(merged.detections) == 1
    assert merged.detections[0].confidence == 0.9  # highest wins
    assert merged.summary["pre_merge_detections"] == 2
    assert merged.summary["frames_processed"] == 4


def test_merge_keeps_distinct_detections():
    a = RunOutput(run_id="a", start_time="s", end_time="e",
                  detections=[_det(1.0, [0, 0, 10, 10])])
    b = RunOutput(run_id="b", start_time="s", end_time="e",
                  detections=[_det(2.0, [200, 200, 10, 10], cls="person")])
    merged = merge_run_outputs([a, b])
    assert len(merged.detections) == 2


def test_merge_is_deterministic(video):
    orch = ParentOrchestrator()
    first = orch.run(_ri(video), windows=3, overlap=0.2)
    second = orch.run(_ri(video), windows=3, overlap=0.2)
    # Detection data (provenance-free canonical view) is byte-identical.
    assert json.dumps(canonical_timeline(first), sort_keys=True) == \
        json.dumps(canonical_timeline(second), sort_keys=True)
    assert first.run_id == second.run_id  # deterministic merged id


def test_partitioned_matches_whole_after_dedupe(video):
    orch = ParentOrchestrator()
    whole = orch.run(_ri(video), windows=1)
    parts = orch.run(_ri(video), windows=3, overlap=0.25)
    # Overlap inflates raw detections but the merge collapses back to the whole set.
    assert parts.summary["pre_merge_detections"] >= len(whole.detections)
    assert len(parts.detections) == len(whole.detections)


def test_empty_merge_raises():
    with pytest.raises(ValueError):
        merge_run_outputs([])


# --------------------------------------------------------------------------- #
# Child worker tile filtering
# --------------------------------------------------------------------------- #
def test_child_tile_filters_detections(video):
    ri = RunInput(video_uri=video, processing_flags={
        "fps": 5, "tile": {"x": 0, "y": 0, "w": 100, "h": 100}})
    out = process_partition(ri)
    assert all(d.bbox[0] + d.bbox[2] / 2 <= 100 for d in out.detections)
    assert out.provenance.extra["tile"]["w"] == 100


# --------------------------------------------------------------------------- #
# Critic / review layer
# --------------------------------------------------------------------------- #
def test_human_sampling_deterministic(video):
    out = ParentOrchestrator().run(_ri(video), windows=1)
    s1 = sample_for_human_review(out, fraction=0.2, seed=7)
    s2 = sample_for_human_review(out, fraction=0.2, seed=7)
    assert [d.model_dump() for d in s1] == [d.model_dump() for d in s2]
    assert 0 < len(s1) <= len(out.detections)


def test_secondary_validation_reports_agreement(video):
    out = ParentOrchestrator().run(_ri(video), windows=1)
    report = secondary_model_validation(_ri(video), out)
    assert report["agreement_ratio"] == 1.0  # same deterministic mock → full agreement
    assert report["primary_count"] == len(out.detections)


def test_review_optional_sections(video):
    out = ParentOrchestrator().run(_ri(video), windows=1)
    minimal = review(_ri(video), out)
    assert "human_review_sample" not in minimal
    full = review(_ri(video), out, human_sample_fraction=0.1, run_secondary=True)
    assert "human_review_sample" in full and "secondary_validation" in full
