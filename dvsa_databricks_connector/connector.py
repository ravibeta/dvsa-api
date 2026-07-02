"""Main connector API: Delta <-> DVSA ``context`` mapping and batch inference.

This module is split into two layers so the important logic stays testable
without a Spark cluster:

* **Pure mapping/batching helpers** (:func:`row_to_context`,
  :func:`result_to_row`, :func:`iter_batches`, :func:`run_inference_on_rows`)
  operate on plain Python dicts/lists and require neither ``pyspark`` nor a live
  DVSA endpoint. They are what the unit tests exercise directly.
* **Spark entry points** (:func:`ingest_videos_to_delta`,
  :func:`prepare_context_from_delta`, :func:`run_inference_batch`) wrap the pure
  helpers around ``pyspark`` DataFrames and Delta I/O. ``pyspark`` is imported
  lazily inside these functions.

The DVSA ``context`` schema (per the integration spec) is::

    {
      "frames":               [ {..frame metadata / base64 thumb..}, ... ],  # optional
      "tracks":               [ {"id", "bbox", "velocity", "timestamp"}, ...],
      "sensor_meta":          {"gps", "altitude", "camera", ...},
      "precomputed_features": {..optional numeric features..},               # optional
      "query":                "optional natural-language task"               # optional
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .client import DVSAClient, build_client
from .config import DVSAConfig

logger = logging.getLogger(__name__)

# Delta column names the mapping looks for. Kept as constants so notebooks and
# the mapping never drift; unknown columns are ignored, missing ones default.
COL_FRAMES = "frames"
COL_TRACKS = "tracks"
COL_SENSOR_META = "sensor_meta"
COL_FEATURES = "precomputed_features"
COL_QUERY = "query"

# Identity columns copied through onto the result row for correlation.
IDENTITY_COLS = ("video_id", "analysis_id", "frame_index", "frame_id")

# The canonical set of columns the inference-results Delta table carries.
RESULT_COLUMNS = (
    "video_id",
    "frame_index",
    "model_name",
    "result_json",
    "num_detections",
    "error",
)


# --------------------------------------------------------------------------- #
# Pure mapping helpers (no pyspark, no network) — the tested core
# --------------------------------------------------------------------------- #
def _coerce_json(value: Any) -> Any:
    """Normalise a Delta cell into a plain Python object.

    Delta rows may deliver a column as a native ``dict``/``list`` (struct/array
    types) or as a JSON **string** (when the column is stored as text). This
    accepts either — plus ``None`` — and returns the parsed object, so the rest
    of the mapping does not care how the table was typed.
    """

    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            # Not JSON — treat as an opaque scalar string (e.g. a query).
            return value
    return value


def row_to_context(
    row: Dict[str, Any], *, include_frames: bool = True
) -> Dict[str, Any]:
    """Map one Delta row (as a dict) to a DVSA ``context`` dict.

    Only the keys DVSA understands are emitted, and empty/optional sections are
    dropped so the payload stays small. ``include_frames=False`` omits the
    (potentially heavy) ``frames`` list — useful when only tracks/features drive
    inference and thumbnails would bloat the request.

    Parameters
    ----------
    row:
        A mapping of column name -> value (e.g. ``pyspark.sql.Row.asDict()``).
    include_frames:
        Whether to include the ``frames`` section.
    """

    context: Dict[str, Any] = {}

    if include_frames:
        frames = _coerce_json(row.get(COL_FRAMES))
        if frames:
            context["frames"] = frames if isinstance(frames, list) else [frames]

    tracks = _coerce_json(row.get(COL_TRACKS))
    if tracks:
        context["tracks"] = tracks if isinstance(tracks, list) else [tracks]

    sensor_meta = _coerce_json(row.get(COL_SENSOR_META))
    if sensor_meta:
        context["sensor_meta"] = sensor_meta

    features = _coerce_json(row.get(COL_FEATURES))
    if features:
        context["precomputed_features"] = features

    query = row.get(COL_QUERY)
    if query:
        context["query"] = query

    return context


def _count_detections(result: Dict[str, Any]) -> int:
    """Best-effort count of detections in a DVSA result payload.

    DVSA results vary by routine/model; we look for the common shapes without
    assuming any single one, defaulting to 0 when none is present.
    """

    if not isinstance(result, dict):
        return 0
    for key in ("detections", "tracks", "results"):
        val = result.get(key)
        if isinstance(val, list):
            return len(val)
    summary = result.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("count"), (int, float)):
        return int(summary["count"])
    return 0


def result_to_row(
    row: Dict[str, Any],
    result: Dict[str, Any],
    model_name: str,
    *,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one ``inference_results`` row from an input row + DVSA result.

    Identity columns (``video_id``/``frame_index``) are copied through for
    correlation; the full DVSA payload is stored as ``result_json`` (a JSON
    string, so the table has a stable, engine-agnostic schema).
    """

    out: Dict[str, Any] = {
        "video_id": row.get("video_id"),
        "frame_index": row.get("frame_index"),
        "model_name": model_name,
        "result_json": json.dumps(result, default=str) if result is not None else None,
        "num_detections": _count_detections(result or {}),
        "error": error,
    }
    return out


def iter_batches(rows: Sequence[Any], batch_size: int) -> Iterator[List[Any]]:
    """Yield ``rows`` in lists of at most ``batch_size`` items."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: List[Any] = []
    for item in rows:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_inference_on_rows(
    rows: Iterable[Dict[str, Any]],
    model_name: str,
    client: DVSAClient,
    *,
    batch_size: int = 64,
    include_frames: bool = True,
) -> Dict[str, Any]:
    """Run DVSA inference over an iterable of row-dicts (pure, no Spark).

    This is the heart of :func:`run_inference_batch`, factored out so it can be
    unit-tested with a stub client and no Spark session. A failing single-row
    inference is captured as an ``error`` on that result row rather than
    aborting the whole batch (so one bad frame does not sink a job).

    Returns
    -------
    dict
        ``{"results": [row, ...], "metrics": {"rows", "batches", "errors",
        "detections"}}``.
    """

    rows = list(rows)
    results: List[Dict[str, Any]] = []
    errors = 0
    detections = 0
    batches = 0

    for batch in iter_batches(rows, batch_size):
        batches += 1
        contexts = [row_to_context(r, include_frames=include_frames) for r in batch]
        for row, context in zip(batch, contexts):
            try:
                result = client.infer(context, model_name)
                out = result_to_row(row, result, model_name)
                detections += out["num_detections"]
            except Exception as exc:  # noqa: BLE001 - isolate per-row failures
                errors += 1
                out = result_to_row(row, {}, model_name, error=str(exc))
            results.append(out)

    metrics = {
        "rows": float(len(rows)),
        "batches": float(batches),
        "errors": float(errors),
        "detections": float(detections),
    }
    return {"results": results, "metrics": metrics}


# --------------------------------------------------------------------------- #
# Spark entry points (lazy pyspark) — the Databricks-facing surface
# --------------------------------------------------------------------------- #
def ingest_videos_to_delta(
    spark: Any,
    source_path: str,
    table: str,
    *,
    fmt: str = "json",
    auto_loader: bool = False,
    checkpoint_location: Optional[str] = None,
    mode: str = "append",
    multiline: bool = True,
) -> Any:
    """Ingest drone video/telemetry metadata files into a Delta table.

    Two ingestion styles, mirroring the youtube-video-intelligence pattern:

    * **batch** (default): a one-shot ``spark.read`` of ``source_path`` written
      to ``table`` as Delta.
    * **Auto Loader** (``auto_loader=True``): a Structured Streaming
      ``cloudFiles`` read that incrementally picks up new files; requires
      ``checkpoint_location``. Returns the ``StreamingQuery``.

    Parameters
    ----------
    spark:
        Active ``SparkSession``.
    source_path:
        Directory/glob of JSON (or ``fmt``) files with frame/track metadata.
    table:
        Target Delta table name (or path) to write to.
    fmt:
        Source file format (``"json"``, ``"parquet"``, ...).
    auto_loader:
        Use Databricks Auto Loader (``cloudFiles``) for incremental ingest.
    checkpoint_location:
        Required when ``auto_loader=True``.
    mode:
        Write mode for batch ingest (``"append"``/``"overwrite"``).
    """

    if auto_loader:
        if not checkpoint_location:
            raise ValueError("auto_loader=True requires checkpoint_location")
        reader = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", fmt)
            .option("multiLine", str(multiline).lower())
        )
        stream = reader.load(source_path)
        return (
            stream.writeStream.format("delta")
            .option("checkpointLocation", checkpoint_location)
            .toTable(table)
        )

    reader = spark.read.format(fmt)
    if fmt == "json":
        reader = reader.option("multiLine", str(multiline).lower())
    df = reader.load(source_path)
    (df.write.format("delta").mode(mode).saveAsTable(table))
    logger.info("Ingested %s into Delta table %s", source_path, table)
    return df


def prepare_context_from_delta(
    spark: Any,
    table: str,
    *,
    where: Optional[str] = None,
    limit: Optional[int] = None,
) -> Any:
    """Read a Delta table and return a DataFrame ready for inference.

    The connector maps whatever context columns are present
    (``frames``/``tracks``/``sensor_meta``/``precomputed_features``/``query``
    plus identity columns); this helper just loads/filters the table. Callers
    can ``.select`` down further before :func:`run_inference_batch`.
    """

    df = spark.read.table(table)
    if where:
        df = df.filter(where)
    if limit is not None:
        df = df.limit(int(limit))
    return df


def run_inference_batch(
    df: Any,
    model_name: str,
    *,
    config: Optional[DVSAConfig] = None,
    client: Optional[DVSAClient] = None,
    batch_size: int = 64,
    output_table: Optional[str] = None,
    output_path: Optional[str] = None,
    write_mode: str = "append",
    include_frames: bool = True,
    mlflow_run: bool = True,
    infer_fn: Any = None,
) -> Any:
    """Run batch DVSA inference over a Spark DataFrame and write Delta results.

    Splits the input into batches, maps each row to a DVSA ``context``, calls
    the DVSA endpoint (remote) or adapter (in-cluster), collects results, writes
    them to the ``inference_results`` Delta table/path, and optionally logs
    metrics to MLflow.

    Parameters
    ----------
    df:
        Input ``pyspark.sql.DataFrame`` (from :func:`prepare_context_from_delta`).
    model_name:
        DVSA model/catalog id to run.
    config:
        :class:`~dvsa_databricks_connector.config.DVSAConfig`. Required unless a
        pre-built ``client`` is supplied.
    client:
        Pre-built :class:`~dvsa_databricks_connector.client.DVSAClient`
        (bypasses ``config``); handy for tests and custom transports.
    batch_size:
        Rows per inference batch.
    output_table / output_path:
        Where to write results as Delta (table name or path). If neither is
        given, results are returned without being persisted.
    include_frames:
        Pass-through to :func:`row_to_context`.
    mlflow_run:
        Log run params/metrics to MLflow (best-effort; skipped if mlflow absent).
    infer_fn:
        In-cluster inference callable, forwarded to :func:`build_client`.

    Returns
    -------
    pyspark.sql.DataFrame
        The ``inference_results`` DataFrame (also written to Delta if a
        destination was given).
    """

    if client is None:
        if config is None:
            raise ValueError("Provide either a config or a pre-built client")
        client = build_client(config, infer_fn=infer_fn)

    spark = df.sparkSession

    # Collect to the driver in row-dict form. For very large inputs, callers
    # should partition upstream or use stream_inference (foreachBatch); this
    # template favours clarity over maximal throughput.
    rows = [r.asDict(recursive=True) for r in df.collect()]
    outcome = run_inference_on_rows(
        rows, model_name, client,
        batch_size=batch_size, include_frames=include_frames,
    )
    results = outcome["results"]
    metrics = outcome["metrics"]

    result_df = _results_to_spark_df(spark, results)

    if output_table or output_path:
        writer = result_df.write.format("delta").mode(write_mode)
        if output_path:
            writer.save(output_path)
        else:
            writer.saveAsTable(output_table)
        logger.info(
            "Wrote %d inference result(s) to %s",
            len(results), output_table or output_path,
        )

    if mlflow_run:
        _log_run_best_effort(model_name, metrics, batch_size,
                             output_table or output_path)

    return result_df


def _results_to_spark_df(spark: Any, results: List[Dict[str, Any]]) -> Any:
    """Build a Spark DataFrame with the stable RESULT_COLUMNS schema."""

    from pyspark.sql.types import (
        IntegerType, LongType, StringType, StructField, StructType,
    )

    schema = StructType([
        StructField("video_id", LongType(), True),
        StructField("frame_index", LongType(), True),
        StructField("model_name", StringType(), True),
        StructField("result_json", StringType(), True),
        StructField("num_detections", IntegerType(), True),
        StructField("error", StringType(), True),
    ])
    # Build ordered tuples (not dicts): Spark maps list-of-dict rows by value
    # order rather than field name, which can silently misalign columns; tuples
    # in the schema's field order are unambiguous across Spark versions.
    rows = [
        (
            _as_long(r.get("video_id")),
            _as_long(r.get("frame_index")),
            r.get("model_name"),
            r.get("result_json"),
            int(r.get("num_detections") or 0),
            r.get("error"),
        )
        for r in results
    ]
    return spark.createDataFrame(rows, schema=schema)


def _as_long(value: Any) -> Optional[int]:
    """Coerce an identity value to int or None (Delta long column)."""

    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _log_run_best_effort(
    model_name: str, metrics: Dict[str, Any], batch_size: int, dest: Optional[str]
) -> None:
    """Log a run to MLflow if available; never fail the pipeline if it isn't."""

    try:
        from .mlflow_utils import log_inference_run

        log_inference_run(
            params={
                "model_name": model_name,
                "batch_size": batch_size,
                "output": dest,
            },
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001 - MLflow is optional
        logger.warning("Skipping MLflow logging: %s", exc)
