"""Integration/trigger tests — signature verification, payload conversion, dispatch.

Fully offline: the default in-process dispatcher runs the deterministic pipeline; no
real Slack/GitHub/webhook traffic is used. Signatures are computed with test secrets.
"""

import json
import os
from urllib.parse import urlencode

import pytest

from agent_kits.integrations.adapters.slack_app.slack_trigger import (
    compute_slack_signature,
    handle_slack_command,
    parse_command_text,
    slack_payload_to_run_input,
    verify_slack_signature,
    SIGNATURE_HEADER as SLACK_SIG,
    TIMESTAMP_HEADER as SLACK_TS,
)
from agent_kits.integrations.adapters.webhook_receiver import (
    SIGNATURE_HEADER as WH_SIG,
    compute_signature,
    handle_webhook,
    verify_webhook_signature,
    webhook_payload_to_run_input,
)
from agent_kits.integrations.run_tracker import RunTracker, make_default_dispatcher
from agent_kits.test_assets.generate_synthetic_video import ensure_manifest

_ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "test_assets", "sample_short.json")
_SECRET = "test-secret"


@pytest.fixture(scope="module")
def video() -> str:
    return ensure_manifest(_ASSET)


# --------------------------------------------------------------------------- #
# Webhook receiver
# --------------------------------------------------------------------------- #
def test_webhook_signature_roundtrip():
    body = b'{"video_uri":"x"}'
    sig = compute_signature(body, _SECRET)
    assert verify_webhook_signature(body, sig, secret=_SECRET)
    assert not verify_webhook_signature(body, "sha256=bad", secret=_SECRET)
    assert not verify_webhook_signature(body, sig, secret="wrong")


def test_webhook_missing_secret_fails_closed():
    body = b"{}"
    assert not verify_webhook_signature(body, "sha256=x", secret="")


def test_webhook_payload_conversion(video):
    ri = webhook_payload_to_run_input(
        {"video_uri": video, "processing_flags": {"fps": 5}, "sensor_id": "cam1"})
    assert ri.video_uri == video and ri.sensor_id == "cam1"


def test_webhook_payload_missing_video():
    with pytest.raises(ValueError):
        webhook_payload_to_run_input({"nope": 1})


def test_handle_webhook_end_to_end(video):
    tracker = RunTracker()
    body = json.dumps({"video_uri": video, "processing_flags": {"fps": 5}}).encode()
    headers = {WH_SIG: compute_signature(body, _SECRET)}
    status, resp = handle_webhook(headers, body, secret=_SECRET, tracker=tracker)
    assert status == 202 and resp["accepted"] is True
    assert resp["run"]["status"] == "completed"
    assert tracker.get(resp["run"]["run_id"])["history"] == \
        ["accepted", "running", "completed"]


def test_handle_webhook_rejects_bad_signature(video):
    body = json.dumps({"video_uri": video}).encode()
    status, resp = handle_webhook({WH_SIG: "sha256=bad"}, body, secret=_SECRET)
    assert status == 401 and "error" in resp


def test_handle_webhook_bad_json(video):
    body = b"not-json"
    headers = {WH_SIG: compute_signature(body, _SECRET)}
    status, resp = handle_webhook(headers, body, secret=_SECRET)
    assert status == 400


# --------------------------------------------------------------------------- #
# Slack slash command
# --------------------------------------------------------------------------- #
def test_slack_signature_roundtrip():
    body = b"text=file://x"
    ts = "1700000000"
    sig = compute_slack_signature(ts, body, _SECRET)
    assert verify_slack_signature(ts, body, sig, secret=_SECRET, now=1700000000)
    # Stale timestamp (replay) rejected.
    assert not verify_slack_signature(ts, body, sig, secret=_SECRET, now=1700000000 + 999)


def test_parse_command_text():
    parsed = parse_command_text("file://vid.json fps=5 sensor_id=cam1 tiles=2")
    assert parsed["video_uri"] == "file://vid.json"
    assert parsed["processing_flags"]["fps"] == 5.0
    assert parsed["processing_flags"]["tiles"] == "2"
    assert parsed["sensor_id"] == "cam1"


def test_parse_command_text_requires_uri():
    with pytest.raises(ValueError):
        parse_command_text("")


def test_slack_payload_to_run_input():
    ri = slack_payload_to_run_input(
        {"text": "file://vid.json fps=3", "user_id": "U123"})
    assert ri.video_uri == "file://vid.json"
    assert ri.processing_flags["fps"] == 3.0
    assert ri.agent_id == "slack:U123"


def test_handle_slack_command_end_to_end(video):
    import time
    tracker = RunTracker()
    body = urlencode({"text": f"{video} fps=5", "user_id": "U9"}).encode()
    ts = str(int(time.time()))  # fresh timestamp so signature is within the window
    headers = {SLACK_TS: ts, SLACK_SIG: compute_slack_signature(ts, body, _SECRET)}
    status, resp = handle_slack_command(
        headers, body, secret=_SECRET,
        dispatcher=make_default_dispatcher(tracker))
    assert status == 200
    assert resp["response_type"] == "in_channel"
    assert resp["run"]["status"] == "completed"


def test_handle_slack_command_bad_signature(video):
    body = urlencode({"text": video}).encode()
    status, resp = handle_slack_command(
        {SLACK_TS: "1700000000", SLACK_SIG: "v0=bad"}, body, secret=_SECRET)
    assert status == 401


def test_handle_slack_command_usage_on_empty(video):
    import time
    body = urlencode({"text": "", "user_id": "U1"}).encode()
    ts = str(int(time.time()))  # fresh timestamp so signature is within the window
    headers = {SLACK_TS: ts, SLACK_SIG: compute_slack_signature(ts, body, _SECRET)}
    status, resp = handle_slack_command(headers, body, secret=_SECRET)
    # Empty text → usage hint (still a 200 so Slack shows the message).
    assert status == 200 and "Usage" in resp["text"]
