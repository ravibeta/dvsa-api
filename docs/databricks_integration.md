# DVSA on Databricks — integration guide

Run end-to-end drone video inference pipelines that embed DVSA analytics
directly inside Databricks — notebooks, Jobs, Workflows and Structured
Streaming — with **no changes to DVSA-APIs core**. The integration ships as an
additive package (`dvsa_databricks_connector/`), notebook templates
(`notebooks/databricks/`), a Job template (`jobs/job_template.json`) and a Delta
pipeline template (`examples/pipeline_template/`).

The design borrows the medallion/pipeline patterns from
`databricks-solutions/youtube-video-intelligence`:
`raw_frames → tracks → features → inference_results`, with MLflow tracking.

---

## 1. Quick start

1. **Install the connector** (notebook cell):

   ```python
   %pip install "git+https://github.com/ravibeta/DVSA-APIs#subdirectory=dvsa_databricks_connector"
   dbutils.library.restartPython()
   ```

   Or cluster-wide via a wheel + the init script
   `dvsa_databricks_connector/cluster-init/install_dvsa_connector.sh`.

2. **Create secrets** (Databricks CLI, run once):

   ```bash
   databricks secrets create-scope dvsa
   databricks secrets put-secret dvsa DVSA_ENDPOINT   # https://dvsa.example.com/api/v1
   databricks secrets put-secret dvsa DVSA_API_KEY
   ```

3. **Run the notebooks** in order — open `notebooks/databricks/01-ingest-and-prepare.ipynb`,
   set the single **CONFIG** cell, and run all. Then `02-batch-inference.ipynb`.

That's it — results land in the `inference_results` Delta table and a run
appears in MLflow.

---

## 2. Configuration

```python
from dvsa_databricks_connector import config_from_secrets
cfg = config_from_secrets(dbutils, scope="dvsa")   # validated DVSAConfig
```

`config_from_secrets` reads `DVSA_ENDPOINT` / `DVSA_API_KEY` (and optional
`DVSA_MODE`) from the scope. Off-cluster (local dev / CI) it falls back to the
`DVSA_ENDPOINT` / `DVSA_API_KEY` **environment variables** so the same cell
works everywhere. `cfg.redacted()` prints the config with the key masked.

### Modes

| Mode | When | How |
| --- | --- | --- |
| `remote` (default) | Call a hosted DVSA-API endpoint. | Set `DVSA_ENDPOINT` + `DVSA_API_KEY`. |
| `in_cluster` | Private deployment; inference runs on the cluster. | Install a DVSA adapter wheel and pass `infer_fn=callable(context, model_name) -> dict`. |

---

## 3. The `context` mapping

`row_to_context` maps a Delta row to the DVSA `context` dict. Columns may be
native struct/array types **or** JSON strings — both are accepted.

| Delta column | context key | Notes |
| --- | --- | --- |
| `frames` | `frames` | optional; list of frame-metadata dicts / base64 thumbs |
| `tracks` | `tracks` | list of `{id, bbox, velocity, timestamp}` |
| `sensor_meta` | `sensor_meta` | `{gps, altitude, camera, ...}` |
| `precomputed_features` | `precomputed_features` | optional numeric features |
| `query` | `query` | optional natural-language task |
| `video_id`, `frame_index` | — | copied onto results for correlation |

The `inference_results` table schema is stable:
`video_id, frame_index, model_name, result_json, num_detections, error`.

---

## 4. Batch inference

```python
from dvsa_databricks_connector import prepare_context_from_delta, run_inference_batch

df = prepare_context_from_delta(spark, "drone.tracks")
results = run_inference_batch(
    df, model_name="visdrone-yolov8x", config=cfg,
    batch_size=64, output_table="drone.inference_results", mlflow_run=True,
)
```

`run_inference_batch` splits input into batches, maps each row to a `context`,
calls DVSA (remote or in-cluster), writes results to Delta, and logs
params/metrics to MLflow. A per-row failure is captured in that row's `error`
column rather than aborting the job.

---

## 5. Streaming inference

```python
from dvsa_databricks_connector import stream_inference

query = stream_inference(
    spark,
    input_path="/Volumes/drone/landing/frames",
    model_name="visdrone-yolov8x",
    checkpoint_location="/Volumes/drone/_chk",
    output_path="/Volumes/drone/inference_results",
    config=cfg,
    batch_size=64,
    max_files_per_trigger=100,   # back-pressure
)
```

Uses Auto Loader (`cloudFiles`) and `foreachBatch` so each micro-batch reuses
the same batch inference path (and the same pooled DVSA client).

---

## 6. Deploy as a Job / Workflow

Edit `jobs/job_template.json` (cluster spec, notebook path, defaults) and:

```bash
databricks jobs create --json @jobs/job_template.json
```

The Job passes `input_delta_path`, `output_delta_path`, `model_name`,
`batch_size` as notebook parameters; the notebooks read them from
`dbutils.widgets`, so they run unchanged. `notebooks/databricks/05-export-to-repo.ipynb`
shows the `dbx` CI/CD flow.

---

## 7. MLflow

```python
from dvsa_databricks_connector import log_to_mlflow, log_inference_run
```

`run_inference_batch(mlflow_run=True)` logs run params (model_name, batch_size,
output) and metrics (rows, batches, errors, detections). `log_to_mlflow` /
`log_inference_run` are best-effort — if MLflow is unavailable they no-op rather
than fail the pipeline. Attach small JSON artifacts (sample input/output,
reasoning-model metadata) via the `artifacts=` argument.

---

## 8. Security & credentials

- Store **only** the DVSA endpoint URL and API key in Databricks Secrets; read
  them at runtime via `config_from_secrets(dbutils)`. Secret values are
  auto-redacted in notebook output.
- Never write the key to logs — the connector logs URLs and retry reasons only,
  and `DVSAConfig.redacted()` masks the key.
- Prefer Unity Catalog governance for the Delta tables; use cluster-scoped init
  scripts (`cluster-init/`) to install private wheels rather than baking keys
  into images.
- For in-cluster mode, distribute the DVSA adapter as a private wheel via the
  init script's `DVSA_ADAPTER_WHEEL` env var.

---

## 9. Best practices

- **Cluster sizing / autoscaling:** the Job template autoscales 2–8 workers;
  size to your frame volume and the DVSA endpoint's throughput.
- **Retry/backoff:** remote calls retry 429/5xx/timeouts with exponential
  backoff (`max_retries`, `backoff_factor` on `DVSAConfig`).
- **Cost controls:** use `max_files_per_trigger` and a scheduled (not
  continuous) trigger for bursty workloads; keep `frames` out of the context
  (`include_frames=False`) when tracks/features suffice.
- **Correlation:** keep `video_id`/`frame_index` on every row so results join
  back to source frames.

---

## 10. Local development & CI

Emulate a Job locally with a local Spark session (offline mock DVSA client):

```bash
python scripts/dbx_local_run.py \
  --input notebooks/sample_data/tracks.json --output /tmp/dvsa_out --format json
```

CI (`.github/workflows/ci-databricks.yml`) installs
`requirements/databricks-dev.txt`, runs the connector tests on local Spark with
mocked DVSA endpoints, lints the package and notebooks, and validates the Job
template — **no live Databricks required**. Run the suite locally with:

```bash
pip install -r requirements/databricks-dev.txt
pip install -e ./dvsa_databricks_connector
pytest -c tests/databricks/pytest.ini tests/databricks
```
