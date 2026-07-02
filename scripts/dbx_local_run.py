#!/usr/bin/env python
"""Emulate a Databricks batch inference Job locally with a local Spark session.

For development only: reads frame/track JSON, runs the *same*
``run_inference_batch`` code path the notebooks/Jobs use, and writes results —
without a Databricks workspace. By default it uses a **mock** in-cluster DVSA
client so the whole thing runs fully offline; pass ``--remote`` to call a real
DVSA endpoint configured via ``DVSA_ENDPOINT`` / ``DVSA_API_KEY``.

Examples
--------
    # Fully offline smoke run against the bundled sample data:
    python scripts/dbx_local_run.py --input notebooks/sample_data/tracks.json \
        --output /tmp/dvsa_out --format json

    # Against a real endpoint (reads DVSA_* env vars):
    python scripts/dbx_local_run.py --input data/tracks.json --remote \
        --model visdrone-yolov8x --output /tmp/dvsa_out
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict


def _mock_infer(context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Deterministic offline DVSA response derived from the context's tracks."""

    tracks = context.get("tracks", []) or []
    return {
        "model_name": model_name,
        "detections": [
            {"id": t.get("id"), "label": t.get("label", "object"), "score": 0.9}
            for t in tracks
        ],
        "summary": {"count": len(tracks)},
    }


def build_spark(app_name: str = "dvsa-local-run") -> Any:
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to frame/track JSON")
    parser.add_argument("--output", required=True, help="Output dir for results")
    parser.add_argument("--model", default="visdrone-yolov8x", help="DVSA model name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--format", default="json", help="Input file format")
    parser.add_argument("--out-format", default="json",
                        help="Output format (json/parquet; avoids needing delta locally)")
    parser.add_argument("--remote", action="store_true",
                        help="Call a real DVSA endpoint (DVSA_* env) instead of the mock")
    args = parser.parse_args(argv)

    from dvsa_databricks_connector import run_inference_batch
    from dvsa_databricks_connector.client import InClusterDVSAClient, build_client
    from dvsa_databricks_connector.config import config_from_env

    spark = build_spark()
    reader = spark.read.format(args.format)
    if args.format == "json":
        reader = reader.option("multiLine", "false")  # sample data is JSON-lines
    df = reader.load(args.input)
    print(f"[dbx-local] read {df.count()} row(s) from {args.input}")

    if args.remote:
        client = build_client(config_from_env().validate())
        print("[dbx-local] using REMOTE DVSA client")
    else:
        client = InClusterDVSAClient(infer_fn=_mock_infer)
        print("[dbx-local] using MOCK in-cluster DVSA client (offline)")

    results = run_inference_batch(
        df,
        model_name=args.model,
        client=client,
        batch_size=args.batch_size,
        mlflow_run=False,  # no tracking server locally
    )
    results.show(truncate=80)
    results.write.mode("overwrite").format(args.out_format).save(args.output)
    print(f"[dbx-local] wrote results to {args.output} ({args.out_format})")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
