# Kit 2 — Cloud Autonomous

**Purpose.** Run drone-video processing unattended in containers and Kubernetes,
exposed as a small HTTP service with metrics, retries, and graceful shutdown.

## Architecture

```
   HTTP client ─▶ FastAPI app (create_app)
                     │  POST /run   ─┐
                     │  GET  /metrics │   execute_run()  ── retries ──▶ run_pipeline
                     │  GET  /healthz │   (framework-independent core) ──▶ RunOutput
                     │  GET  /readyz  ┘        │
                     └── RunnerMetrics ◀───────┘  (Prometheus text)
```

The execution **core** (`execute_run`, `_with_retries`, `RunnerMetrics`) has no web
dependency and is unit-tested directly; FastAPI is a thin, lazily-imported layer.

## Quickstart

```bash
pip install -r agent_kits/requirements.txt uvicorn

# Serve locally
uvicorn agent_kits.cloud_autonomous.cloud_runner:app --port 8080

# Submit a run
curl -s localhost:8080/run -H 'content-type: application/json' \
  -d '{"video_uri":"file:///abs/path/sample_short.json","processing_flags":{"fps":5}}' | jq .

# Metrics & health
curl -s localhost:8080/metrics
curl -s localhost:8080/healthz
```

### Container

```bash
docker build -f agent_kits/cloud_autonomous/Dockerfile -t dvsa-agent-runner:0.1.0 .
docker run -p 8080:8080 --env-file agent_kits/cloud_autonomous/secrets/.env dvsa-agent-runner:0.1.0
```

### Kubernetes

```bash
kubectl create secret generic dvsa-runner-secrets \
  --from-literal=DVSA_API_KEY=... --from-literal=MODEL_ENDPOINT=...
kubectl apply -f agent_kits/cloud_autonomous/k8s/agent-runner-deployment.yaml
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `DVSA_API_KEY` | Bearer token required on `POST /run`. Unset ⇒ auth disabled (dev only). |
| `MODEL_ENDPOINT` | External model endpoint (mocked in tests). |
| `STORAGE_BUCKET` | Output target (`s3://`, `gs://`, `file://`, or local dir). |
| `RUNNER_MAX_PARALLELISM` | Max concurrent runs per instance (default 4). |
| `RUNNER_RETRY_ATTEMPTS` / `RUNNER_RETRY_BACKOFF_S` | Retry policy per run. |
| `MODEL_VERSION` | Pinned model version recorded in provenance. |

Full list: [`secrets/.env.example`](secrets/.env.example). **No secrets are
committed** — only variable names are documented.

## Sample output (`POST /run` → 200)

```json
{"run_id":"run_1f0a…","agent_id":"cloud-runner","start_time":"…","end_time":"…",
 "model_version":"mock-detector-1.0.0","detections":[...],
 "summary":{"total_detections":33,"by_class":{"car":16,"person":16},
            "frames_processed":15}}
```

Status codes: `200` success · `401` bad/missing API key · `422` invalid body ·
`429` at capacity · `503` draining · `500` run error.

## Cost control

- Cap `RUNNER_MAX_PARALLELISM` to bound concurrent inference cost.
- Bound `processing_flags.fps` on each request to limit frames/inferences.
- Autoscale on `dvsa_runner_in_flight` so idle replicas scale to zero/min.

## Observability

See [`observability.md`](observability.md) — metrics catalogue, PromQL, dashboards,
and deployment recommendations. Every run emits `run_id`, `agent_id`, `trace_id`,
`model_version`, `start_time`, `end_time`.

## Extension guide

- **Storage/inference**: swap adapters (S3/GCS/Triton) by passing them into
  `execute_run` or by wiring your own into `run_pipeline`.
- **Auth**: replace the `require_api_key` dependency with your IdP/JWT validation.

## Model-version pinning

Set `MODEL_VERSION` (or configure your real `InferenceAdapter`) to a fully-qualified
tag such as `yolo-v8@2026-07-01`. It is stamped onto every detection for audit.
