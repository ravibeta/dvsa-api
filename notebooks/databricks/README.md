# DVSA on Databricks — notebook templates

Six runnable templates that take a Databricks user from install to a scheduled
Job. Each notebook has a single **CONFIG** cell (widget-backed) as the only edit
you need; everything else runs unchanged.

| Notebook | What it does |
| --- | --- |
| `00-setup.ipynb` | Install the connector, wire Databricks Secrets, set up MLflow. |
| `01-ingest-and-prepare.ipynb` | Ingest sample drone metadata to Delta, prepare context. |
| `02-batch-inference.ipynb` | Batch `run_inference_batch` → Delta + MLflow + viz. |
| `03-streaming-inference.ipynb` | Structured Streaming (`stream_inference`) with Auto Loader. |
| `04-deploy-job.ipynb` | Parameterize + deploy as a Databricks Job. |
| `05-export-to-repo.ipynb` | Add to Databricks Repos and deploy with `dbx`. |

## Quick start (5 steps, no core code changes)

1. Open `00-setup.ipynb`, run the `%pip install` cell, restart Python.
2. Create secrets: `databricks secrets create-scope dvsa` then put
   `DVSA_ENDPOINT` and `DVSA_API_KEY`.
3. Upload `../sample_data/tracks.json` to a Volume/DBFS landing zone (or point
   the config at your own data).
4. Open `01-ingest-and-prepare.ipynb`, set the **CONFIG** cell, run all.
5. Open `02-batch-inference.ipynb`, run all — results land in the
   `inference_results` Delta table and a run appears in MLflow.

Sample data lives in [`../sample_data/`](../sample_data). The connector maps
each Delta row to the DVSA `context` schema documented in
[`../../docs/databricks_integration.md`](../../docs/databricks_integration.md).
