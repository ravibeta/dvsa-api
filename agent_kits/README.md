# DVSA Agent Kits v0.1.0

Four complementary, Warp-style **agent kits** for aerial drone-video analytics.
Each kit runs and extends the same drone-video-analysis pipeline from a different
operating posture, all sharing one deterministic foundation.

| Kit | Directory | Purpose |
| --- | --- | --- |
| **Local Interactive** | [`local_interactive/`](local_interactive/) | Agent-guided local execution & experimentation (CLI + skills, dry-run/confirm). |
| **Cloud Autonomous** | [`cloud_autonomous/`](cloud_autonomous/) | Unattended container/K8s processing (FastAPI runner, metrics, retries). |
| **Orchestration** | [`orchestration/`](orchestration/) | Parent/child partitioned execution + deterministic merge + optional critic. |
| **Integrations & Triggers** | [`integrations/`](integrations/) | Event-driven launches (GitHub Actions, Slack, generic webhooks). |

## Architecture

```
                 ┌────────────────────────── agent_kits.common ──────────────────────────┐
                 │  schemas (RunInput/Detection/RunOutput)  ·  adapters  ·  pipeline       │
                 │  utils (run-id, IoU, dedupe, merge)      ·  structured JSON logging     │
                 └───────────────▲───────────────▲───────────────▲───────────────▲────────┘
                                 │               │               │               │
                    ┌────────────┴───┐  ┌────────┴────────┐  ┌───┴──────────┐  ┌─┴────────────┐
                    │ Local          │  │ Cloud           │  │ Orchestration │  │ Integrations │
                    │ Interactive    │  │ Autonomous      │  │ (parent/child)│  │ & Triggers   │
                    │ cli + harness  │  │ FastAPI runner  │  │ merge + critic│  │ slack/webhook│
                    └────────────────┘  └─────────────────┘  └───────────────┘  └──────────────┘
```

Everything flows through the shared contracts, so an event that arrives via a Slack
slash command is processed identically to one launched from the local CLI.

## Quickstart

```bash
# 1. (optional) create the deterministic synthetic test video
python agent_kits/test_assets/generate_synthetic_video.py --out /tmp/sample.json

# 2. Local Interactive — run the full pipeline with a dry-run first
python -m agent_kits.local_interactive.cli_wrapper run \
    --video file:///tmp/sample.json --dry-run
python -m agent_kits.local_interactive.cli_wrapper run \
    --video file:///tmp/sample.json --out /tmp/out

# 3. Cloud Autonomous — serve locally and POST a run
uvicorn agent_kits.cloud_autonomous.cloud_runner:app --port 8080 &
curl -s localhost:8080/run -H 'content-type: application/json' \
    -d '{"video_uri":"file:///tmp/sample.json"}' | jq .

# 4. Orchestration — partition + merge deterministically
python -m agent_kits.orchestration.parent_orchestrator \
    --video file:///tmp/sample.json --windows 3 --out /tmp/merged.json
```

## Environment variables (overview)

| Variable | Kit | Meaning |
| --- | --- | --- |
| `AGENT_KITS_LOG_LEVEL` | all | Log level (default `INFO`). |
| `DVSA_API_KEY` | cloud | Bearer token required by the cloud runner (unset = auth disabled for local dev). |
| `MODEL_ENDPOINT` | cloud | External model endpoint (mocked in tests). |
| `STORAGE_BUCKET` | cloud/orch | Default output storage target. |
| `RUNNER_MAX_PARALLELISM` | cloud/orch | Max concurrent partitions/workers. |
| `SLACK_SIGNING_SECRET` | integrations | HMAC secret for Slack request verification. |
| `WEBHOOK_SECRET` | integrations | HMAC secret for generic webhook verification. |

See each kit's README for its full variable list. **No secrets are committed** —
all configuration is read from the environment; `.env.example` files document names
only.

## Sample output (`RunOutput`)

```json
{
  "run_id": "run_1f0a3c...",
  "agent_id": "local-interactive",
  "start_time": "2026-07-12T10:00:00+00:00",
  "end_time":   "2026-07-12T10:00:03+00:00",
  "model_version": "mock-detector-1.0.0",
  "detections": [
    {"timestamp": 0.0, "bbox": [10,10,40,30], "class_name": "car",
     "confidence": 0.92, "model_version": "mock-detector-1.0.0",
     "provenance": {"source": "run_pipeline", "trace_id": "…"}}
  ],
  "summary": {"total_detections": 12, "by_class": {"car": 9, "person": 3},
              "frames_processed": 15}
}
```

## Cost control

- **Pin the frame rate**: `processing_flags.fps` bounds frames (and inference calls).
- **Bound parallelism**: `RUNNER_MAX_PARALLELISM` caps concurrent model calls.
- **Partition narrowly**: orchestration `--windows`/tiles let you process only the
  span you need; overlaps are configurable to trade recall for cost.
- **Dry-run first** in the local kit to validate inputs before spending inference.
- **Sample the critic**: secondary-model review runs on a configurable fraction.

## Extension guide

Every IO/inference boundary is a `Protocol` in
[`common/adapters.py`](common/adapters.py). To integrate real infrastructure,
implement the matching interface and pass it in — no core changes required:

```python
class S3Storage:                       # implements StorageAdapter
    def store_output(self, output, dest): ...

class TritonInference:                 # implements InferenceAdapter
    model_version = "yolo-v8@2026-07"
    def run_inference_on_frames(self, frames): ...

run_pipeline(run_input, inference=TritonInference(), storage=S3Storage(),
             store_dest="s3://bucket/runs/")
```

### Option: reuse the existing DVSA views (built-in smarts)

Instead of the offline reference codepaths, a run can leverage the production
`apps/videos/views.py` views in-process — reusing their Azure upload + signal-driven
indexing (`VideoUploadAPIView`) and agentic RAG synthesis (`ChatAPIView`). Django/DRF
are imported lazily, so this stays opt-in and the package remains offline-importable.

```python
from agent_kits.common import DvsaVideoUploadAdapter, DvsaChatAnalyzer, run_pipeline

ingestor = DvsaVideoUploadAdapter(account_id="acct-1", user=request.user)
analyzer = DvsaChatAnalyzer(account_id="acct-1", user=request.user)

out = run_pipeline(run_input,                    # offline detection still runs
                   ingestor=ingestor,            # → VideoUploadAPIView (upload+index)
                   analyzer=analyzer,            # → ChatAPIView (agentic answer)
                   query="what happened at 0:02?")
# out.summary["ingestion"]        -> VideoEntity payload
# out.summary["agentic_answer"]   -> synthesised text
```

`DvsaVideoUploadAdapter` is also a `VideoFetcher` (`ingest_on_fetch=True` ingests
during `fetch_video`). `account_id`/`user` fall back to `DVSA_ACCOUNT_ID`. Both hooks
default off, so omitting them leaves the offline behaviour byte-identical.

## Model-version pinning

Every `Detection` and `RunOutput` carries a `model_version`. Pin it explicitly on
your `InferenceAdapter` (e.g. `"yolo-v8@2026-07-01"`) so outputs are attributable
and reproducible; never rely on a floating "latest" tag. The mock adapter uses
`mock-detector-1.0.0`.

## Testing

```bash
pytest agent_kits -q          # all kits, fully offline & mocked
```

See [`CHANGELOG.md`](CHANGELOG.md), [`SECURITY.md`](SECURITY.md), and the
[migration guide](#migration) in the PR description.
