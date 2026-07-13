"""Smoke tests for the Local Interactive kit (offline, deterministic)."""

import json
import os

import pytest

from agent_kits.common import RunInput, RunOutput
from agent_kits.local_interactive.agent_harness import (
    GatedActionDenied,
    LocalAgentHarness,
    build_run_output,
)
from agent_kits.local_interactive import cli_wrapper
from agent_kits.test_assets.generate_synthetic_video import ensure_manifest

_ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "test_assets", "sample_short.json")


@pytest.fixture(scope="module")
def video() -> str:
    return ensure_manifest(_ASSET)


def _ri(video: str) -> RunInput:
    return RunInput(video_uri=video, processing_flags={"fps": 5})


def test_skills_load_dynamically(video):
    harness = LocalAgentHarness(mode="auto")
    assert set(harness.list_skills()) == {
        "extract_frames", "detect_objects", "summarize_timeline"}


def test_dry_run_executes_nothing(video):
    harness = LocalAgentHarness(mode="dry_run")
    result = harness.run(_ri(video), store_dest="/tmp/should_not_write")
    assert result["executed"] is False
    assert "write_output (always-ask)" in result["planned_steps"]


def test_run_produces_schema_valid_output(video):
    harness = LocalAgentHarness(mode="confirm")
    result = harness.run(_ri(video))
    assert result["executed"] is True
    out = build_run_output(result)
    assert isinstance(out, RunOutput)
    assert out.detections and out.run_id.startswith("run_")
    assert out.summary["frames_processed"] == 15
    # Every detection carries provenance + a pinned model version.
    assert all(d.model_version for d in out.detections)


def test_write_output_gated_denied_without_confirmation(video, tmp_path):
    harness = LocalAgentHarness(mode="confirm", confirm_fn=lambda a, d: False)
    with pytest.raises(GatedActionDenied):
        harness.run(_ri(video), store_dest=str(tmp_path))


def test_write_output_gated_allowed_with_confirmation(video, tmp_path):
    harness = LocalAgentHarness(mode="confirm", confirm_fn=lambda a, d: True)
    result = harness.run(_ri(video), store_dest=str(tmp_path))
    assert result["stored_uri"].startswith("file://")
    # The persisted file exists and is schema-valid JSON.
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    RunOutput.model_validate(data)


def test_skill_summarize_timeline(video):
    harness = LocalAgentHarness(mode="auto")
    result = harness.run_skill("summarize_timeline", _ri(video))
    assert result["summary"]["total_detections"] > 0
    assert "by_class" in result["summary"]


def test_guardrail_blocks_excessive_fps(video):
    harness = LocalAgentHarness(mode="confirm")
    ri = RunInput(video_uri=video, processing_flags={"fps": 999})
    with pytest.raises(ValueError):
        harness.run(ri)


def test_cli_dry_run(video, capsys):
    rc = cli_wrapper.main(["--dry-run", "run", "--video", video, "--fps", "5"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["executed"] is False


def test_cli_run_inference(video, capsys):
    rc = cli_wrapper.main(["run-inference", "--video", video, "--fps", "5"])
    assert rc == 0
    # Logs go to stderr, so stdout is the pure JSON result.
    out = json.loads(capsys.readouterr().out)
    assert out["executed"] is True
    assert isinstance(out["detections"], list) and out["detections"]


def test_cli_write_output_with_yes(video, tmp_path, capsys):
    rc = cli_wrapper.main(
        ["--yes", "write-output", "--video", video, "--fps", "5", "--out", str(tmp_path)])
    assert rc == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
