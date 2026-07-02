"""``dvsa_databricks_connector`` — drop-in Databricks integration for DVSA-APIs.

A small, additive package that lets Databricks users ingest drone
video/telemetry into Delta, map it to the DVSA ``context`` schema, and run
DVSA inference from notebooks, Jobs and Structured Streaming — in either
**remote** mode (hosted DVSA endpoint) or **in-cluster** mode (DVSA adapter
installed on the cluster), with MLflow tracking.

Public API
----------
Config / client:
    :func:`config_from_secrets`, :func:`config_from_env`, :class:`DVSAConfig`,
    :func:`build_client`, :class:`RemoteDVSAClient`, :class:`InClusterDVSAClient`.
Pipeline:
    :func:`ingest_videos_to_delta`, :func:`prepare_context_from_delta`,
    :func:`run_inference_batch`, :func:`stream_inference`.
Mapping (pure, testable):
    :func:`row_to_context`, :func:`result_to_row`, :func:`run_inference_on_rows`.
MLflow:
    :func:`log_to_mlflow`, :func:`log_inference_run`.
"""

from __future__ import annotations

from .client import (
    DVSAClient,
    InClusterDVSAClient,
    RemoteDVSAClient,
    build_client,
)
from .config import (
    DVSAConfig,
    MODE_IN_CLUSTER,
    MODE_REMOTE,
    config_from_env,
    config_from_secrets,
)
from .connector import (
    ingest_videos_to_delta,
    prepare_context_from_delta,
    result_to_row,
    row_to_context,
    run_inference_batch,
    run_inference_on_rows,
)
from .mlflow_utils import log_inference_run, log_to_mlflow, mlflow_available
from .streaming import stream_inference

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "DVSAConfig",
    "config_from_env",
    "config_from_secrets",
    "MODE_REMOTE",
    "MODE_IN_CLUSTER",
    # client
    "DVSAClient",
    "RemoteDVSAClient",
    "InClusterDVSAClient",
    "build_client",
    # connector
    "ingest_videos_to_delta",
    "prepare_context_from_delta",
    "run_inference_batch",
    "run_inference_on_rows",
    "row_to_context",
    "result_to_row",
    # streaming
    "stream_inference",
    # mlflow
    "log_to_mlflow",
    "log_inference_run",
    "mlflow_available",
]
