# dvsa_databricks_connector

A small, additive Python package that makes [DVSA-APIs](https://github.com/ravibeta/DVSA-APIs)
a first-class step inside Databricks — ingest drone video/telemetry into Delta,
map it to the DVSA `context` schema, and run inference from notebooks, Jobs, or
Structured Streaming, with MLflow tracking. **No changes to DVSA-APIs core are
required.**

## Install

From a Databricks notebook (quickest):

```python
%pip install "git+https://github.com/ravibeta/DVSA-APIs#subdirectory=dvsa_databricks_connector"
```

Or build a wheel and install cluster-wide via
[`cluster-init/install_dvsa_connector.sh`](cluster-init/install_dvsa_connector.sh):

```bash
cd dvsa_databricks_connector && python -m build   # produces dist/*.whl
```

Only `requests` is a hard dependency. `pyspark`, Delta and `mlflow` ship with
the Databricks Runtime and are declared as optional extras.

## Configure (Databricks Secrets)

```python
from dvsa_databricks_connector import config_from_secrets
cfg = config_from_secrets(dbutils, scope="dvsa")   # reads DVSA_ENDPOINT / DVSA_API_KEY
```

Create the secrets once with the Databricks CLI:

```bash
databricks secrets create-scope dvsa
databricks secrets put-secret dvsa DVSA_ENDPOINT   # https://dvsa.example.com/api/v1
databricks secrets put-secret dvsa DVSA_API_KEY
```

Locally / in CI, `config_from_secrets(None)` falls back to the `DVSA_ENDPOINT`
and `DVSA_API_KEY` environment variables.

## Use

```python
from dvsa_databricks_connector import (
    prepare_context_from_delta, run_inference_batch, stream_inference,
)

df = prepare_context_from_delta(spark, "drone.tracks")
results = run_inference_batch(df, model_name="visdrone-yolov8x", config=cfg,
                              output_table="drone.inference_results")

# Streaming
q = stream_inference(spark, "/Volumes/drone/frames", "visdrone-yolov8x",
                     checkpoint_location="/Volumes/drone/_chk",
                     output_path="/Volumes/drone/inference_results", config=cfg)
```

### Modes

* **remote** (default) — calls a hosted DVSA-API endpoint.
* **in_cluster** — runs a DVSA adapter installed on the cluster; pass
  `infer_fn=callable(context, model_name) -> dict` to `run_inference_batch`.

## The `context` schema

`row_to_context` maps a Delta row to:

```json
{
  "frames": [ ... ],
  "tracks": [ {"id": 1, "bbox": [x,y,w,h], "velocity": [vx,vy], "timestamp": 0.0} ],
  "sensor_meta": {"gps": [...], "altitude": 120, "camera": {...}},
  "precomputed_features": { ... },
  "query": "count vehicles in the north zone"
}
```

Columns may be native struct/array types **or** JSON strings — both are
accepted. See [`docs/databricks_integration.md`](../docs/databricks_integration.md)
for the full guide.
