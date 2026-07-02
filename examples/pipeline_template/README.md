# DVSA pipeline template (Delta medallion)

A minimal ETL → Feature → Inference → Reporting pipeline that mirrors the
[`databricks-solutions/youtube-video-intelligence`](https://github.com/databricks-solutions)
structure, adapted for drone video analytics.

```
raw_frames ──ETL──▶ tracks ──features──▶ features ──inference──▶ inference_results ──▶ reporting
```

| Stage | Table | Produced by |
| --- | --- | --- |
| ETL | `raw_frames` | `ingest_videos_to_delta()` (Auto Loader / batch) |
| Feature | `tracks`, `features` | notebook 01, feature precompute |
| Inference | `inference_results` | `run_inference_batch()` / `stream_inference()` |
| Reporting | dashboards | `reporting.sql` |

- [`schemas.sql`](schemas.sql) — create the four Delta tables.
- [`reporting.sql`](reporting.sql) — reporting queries over `inference_results`.

See [`../../docs/databricks_integration.md`](../../docs/databricks_integration.md)
for the full walkthrough and [`../../notebooks/databricks/`](../../notebooks/databricks)
for runnable notebooks.
