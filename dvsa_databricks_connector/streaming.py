"""Structured Streaming helpers for continuous DVSA inference.

:func:`stream_inference` wires a Spark Structured Streaming source (new
frame/track files landing in ``input_path``) to DVSA inference, writing results
to a Delta sink. Each micro-batch is handed to
:func:`~dvsa_databricks_connector.connector.run_inference_batch` via
``foreachBatch`` — so the *same* mapping/batching/DVSA-call code path serves
both batch and streaming, and back-pressure is controlled by the trigger and
``max_files_per_trigger``.

``pyspark`` is imported lazily so the module imports without Spark for docs/tests.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .client import DVSAClient, build_client
from .config import DVSAConfig
from .connector import run_inference_batch

logger = logging.getLogger(__name__)


def stream_inference(
    spark: Any,
    input_path: str,
    model_name: str,
    checkpoint_location: str,
    output_path: str,
    *,
    config: Optional[DVSAConfig] = None,
    client: Optional[DVSAClient] = None,
    fmt: str = "json",
    auto_loader: bool = True,
    batch_size: int = 64,
    max_files_per_trigger: Optional[int] = None,
    include_frames: bool = True,
    infer_fn: Any = None,
    await_termination: bool = False,
) -> Any:
    """Continuously infer over newly-arriving frames/tracks and write Delta.

    Parameters
    ----------
    spark:
        Active ``SparkSession``.
    input_path:
        Directory that new frame/track files land in.
    model_name:
        DVSA model/catalog id to run.
    checkpoint_location:
        Structured Streaming checkpoint dir (exactly-once + restart safety).
    output_path:
        Delta path for the ``inference_results`` sink.
    config / client:
        DVSA connection settings or a pre-built client (see
        :func:`run_inference_batch`). Built once and reused across micro-batches.
    fmt:
        Source file format for the stream reader.
    auto_loader:
        Use Databricks Auto Loader (``cloudFiles``) when ``True`` (recommended on
        Databricks); otherwise a plain file-source ``readStream``.
    batch_size:
        Rows per DVSA inference batch *within* each micro-batch.
    max_files_per_trigger:
        Back-pressure knob — cap files ingested per trigger.
    include_frames:
        Pass-through to the mapping.
    infer_fn:
        In-cluster inference callable (in-cluster mode).
    await_termination:
        If ``True``, block until the query terminates (useful in scripts/tests);
        otherwise return the running :class:`StreamingQuery` immediately.

    Returns
    -------
    pyspark.sql.streaming.StreamingQuery
        The running query (already started).
    """

    # Build the DVSA client once so each micro-batch reuses the HTTP session /
    # loaded adapter instead of reconnecting per batch.
    if client is None:
        if config is None:
            raise ValueError("Provide either a config or a pre-built client")
        client = build_client(config, infer_fn=infer_fn)

    reader = _build_reader(
        spark, fmt=fmt, auto_loader=auto_loader,
        max_files_per_trigger=max_files_per_trigger,
    )
    stream_df = reader.load(input_path)

    def _process_micro_batch(batch_df: Any, batch_id: int) -> None:
        # foreachBatch hands us a *batch* DataFrame per micro-batch; reuse the
        # batch inference path with the shared client and append to the sink.
        count = batch_df.count()
        logger.info("Micro-batch %s: %d row(s)", batch_id, count)
        if count == 0:
            return
        run_inference_batch(
            batch_df,
            model_name,
            client=client,
            batch_size=batch_size,
            output_path=output_path,
            write_mode="append",
            include_frames=include_frames,
            # Per-micro-batch MLflow runs would be noisy; log at the job level.
            mlflow_run=False,
        )

    query = (
        stream_df.writeStream
        .foreachBatch(_process_micro_batch)
        .option("checkpointLocation", checkpoint_location)
        .start()
    )
    logger.info(
        "Started streaming inference: %s -> %s (model=%s)",
        input_path, output_path, model_name,
    )
    if await_termination:
        query.awaitTermination()
    return query


def _build_reader(
    spark: Any,
    *,
    fmt: str,
    auto_loader: bool,
    max_files_per_trigger: Optional[int],
) -> Any:
    """Construct the Structured Streaming reader (Auto Loader or file source)."""

    if auto_loader:
        reader = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", fmt)
        )
    else:
        reader = spark.readStream.format(fmt)
    if fmt == "json":
        reader = reader.option("multiLine", "true")
    if max_files_per_trigger is not None:
        reader = reader.option("maxFilesPerTrigger", int(max_files_per_trigger))
    return reader
