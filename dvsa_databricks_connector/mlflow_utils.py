"""MLflow logging helpers for DVSA inference runs.

Wraps the small slice of MLflow the connector needs — starting a run, logging
params/metrics, and attaching small JSON artifacts (sample inputs/outputs,
reasoning-model metadata). Every helper is **best-effort and optional**:
``mlflow`` is imported lazily, and if it is not installed (or logging fails) the
helpers degrade to no-ops rather than breaking a pipeline. This keeps the
connector runnable off-cluster and in CI without an MLflow tracking server.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def mlflow_available() -> bool:
    """Return ``True`` if the ``mlflow`` package can be imported."""

    try:
        import mlflow  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any import error means "not available"
        return False


def log_to_mlflow(
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    *,
    artifacts: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    nested: bool = False,
) -> Optional[str]:
    """Log a single MLflow run and return its ``run_id`` (or ``None``).

    Parameters
    ----------
    metrics:
        Numeric metrics (latency, calls, anomalies, detections, ...).
    params:
        Run params (model_name, model_version, batch_size, ...). Values are
        stringified by MLflow.
    artifacts:
        Mapping of ``filename -> JSON-serialisable object`` written as artifacts
        (e.g. ``{"sample_input.json": {...}, "sample_output.json": {...}}``).
    run_name:
        Optional MLflow run name.
    nested:
        Start a nested run (when already inside an active run).

    Returns
    -------
    Optional[str]
        The MLflow ``run_id``, or ``None`` if MLflow is unavailable.
    """

    if not mlflow_available():
        logger.info("mlflow not available; skipping run logging")
        return None

    import mlflow

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if params:
            mlflow.log_params({k: _stringify(v) for k, v in params.items()})
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in _numeric_only(metrics).items()})
        if artifacts:
            _log_json_artifacts(mlflow, artifacts)
        return run.info.run_id


def log_inference_run(
    *,
    run_id: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
) -> Optional[str]:
    """Log an inference run's params/metrics/artifacts to MLflow.

    Convenience wrapper matching the spec's
    ``log_inference_run(run_id, metrics, params, artifacts)`` signature. If
    ``run_id`` is given, values are logged into that existing run; otherwise a
    new run is started. Returns the effective ``run_id`` (or ``None``).
    """

    if not mlflow_available():
        logger.info("mlflow not available; skipping inference-run logging")
        return None

    import mlflow

    if run_id is not None:
        client = mlflow.tracking.MlflowClient()
        if params:
            for k, v in params.items():
                client.log_param(run_id, k, _stringify(v))
        if metrics:
            for k, v in _numeric_only(metrics).items():
                client.log_metric(run_id, k, float(v))
        if artifacts:
            _log_json_artifacts_to_run(client, run_id, artifacts)
        return run_id

    return log_to_mlflow(metrics, params, artifacts=artifacts, run_name=run_name)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _numeric_only(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Keep only numeric (non-bool) metric values — MLflow requires floats."""

    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _log_json_artifacts(mlflow_mod: Any, artifacts: Dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for name, obj in artifacts.items():
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, default=str, indent=2)
            mlflow_mod.log_artifact(path)


def _log_json_artifacts_to_run(client: Any, run_id: str, artifacts: Dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for name, obj in artifacts.items():
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, default=str, indent=2)
            client.log_artifact(run_id, path)
