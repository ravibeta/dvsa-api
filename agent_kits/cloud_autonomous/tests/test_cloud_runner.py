"""Tests for the Cloud Autonomous runner.

The framework-independent core (retries, metrics, execution) is tested everywhere.
The HTTP layer is exercised via FastAPI's TestClient only where FastAPI is
installed (CI); those tests are skipped otherwise. No network is used — inference is
the deterministic mock adapter.
"""

import os

import pytest

from agent_kits.cloud_autonomous.cloud_runner import (
    RunnerMetrics,
    _with_retries,
    create_app,
    execute_run,
)
from agent_kits.common import RunInput, RunOutput
from agent_kits.test_assets.generate_synthetic_video import ensure_manifest

_ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "test_assets", "sample_short.json")


@pytest.fixture(scope="module")
def video() -> str:
    return ensure_manifest(_ASSET)


# --------------------------------------------------------------------------- #
# Core (framework-independent)
# --------------------------------------------------------------------------- #
def test_execute_run_produces_output(video):
    metrics = RunnerMetrics()
    out = execute_run(
        RunInput(video_uri=video, processing_flags={"fps": 5}), metrics=metrics)
    assert isinstance(out, RunOutput)
    assert out.detections
    assert metrics.runs_total == 1 and metrics.runs_succeeded == 1


def test_retries_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    result, retries = _with_retries(flaky, attempts=3, backoff_s=0)
    assert result == "ok" and retries == 2


def test_retries_exhausted_raises():
    def always_fail():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError):
        _with_retries(always_fail, attempts=2, backoff_s=0)


def test_metrics_render_prometheus():
    metrics = RunnerMetrics()
    metrics.start()
    metrics.finish(ok=True, latency_s=0.1, retries=1)
    text = metrics.render()
    assert "dvsa_runner_runs_total 1" in text
    assert "# TYPE dvsa_runner_runs_total counter" in text
    assert "dvsa_runner_retries_total 1" in text


# --------------------------------------------------------------------------- #
# HTTP layer (FastAPI only — CI)
# --------------------------------------------------------------------------- #
try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient
    _HAS_FASTAPI = True
except Exception:  # noqa: BLE001
    _HAS_FASTAPI = False

fastapi_only = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


@fastapi_only
def test_http_run_ok(video, monkeypatch):
    monkeypatch.delenv("DVSA_API_KEY", raising=False)
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    client = TestClient(create_app())
    resp = client.post("/run", json={"video_uri": video,
                                     "processing_flags": {"fps": 5}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["detections"] and body["run_id"].startswith("run_")


@fastapi_only
def test_http_run_validation_error(video):
    client = TestClient(create_app())
    resp = client.post("/run", json={"processing_flags": {"fps": 5}})  # missing video_uri
    assert resp.status_code == 422


@fastapi_only
def test_http_auth_enforced(video, monkeypatch):
    monkeypatch.setenv("DVSA_API_KEY", "secret-token")
    client = TestClient(create_app())
    unauth = client.post("/run", json={"video_uri": video})
    assert unauth.status_code == 401
    ok = client.post("/run", json={"video_uri": video,
                                   "processing_flags": {"fps": 5}},
                     headers={"Authorization": "Bearer secret-token"})
    assert ok.status_code == 200, ok.text


@fastapi_only
def test_http_metrics_and_health(video):
    client = TestClient(create_app())
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "dvsa_runner_runs_total" in metrics.text
