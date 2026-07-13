# Observability — Cloud Autonomous runner

The runner is built to be observed in production: every run emits structured logs
and updates Prometheus-compatible counters, all correlatable by `trace_id`/`run_id`.

## Metrics (`GET /metrics`)

Prometheus text exposition, scrapeable directly (the K8s deployment sets the
`prometheus.io/scrape` annotations):

| Metric | Type | Meaning |
| --- | --- | --- |
| `dvsa_runner_runs_total` | counter | Runs accepted. |
| `dvsa_runner_runs_succeeded_total` | counter | Runs that completed successfully. |
| `dvsa_runner_runs_failed_total` | counter | Runs that failed after retries. |
| `dvsa_runner_retries_total` | counter | Total retry attempts across runs. |
| `dvsa_runner_in_flight` | gauge | Runs currently executing. |
| `dvsa_runner_latency_seconds_sum` | counter | Cumulative run latency (seconds). |

Useful derived queries:

```promql
# success rate over 5m
rate(dvsa_runner_runs_succeeded_total[5m])
  / clamp_min(rate(dvsa_runner_runs_total[5m]), 1)

# average run latency over 5m
rate(dvsa_runner_latency_seconds_sum[5m])
  / clamp_min(rate(dvsa_runner_runs_succeeded_total[5m]), 1)

# saturation
dvsa_runner_in_flight
```

## Logs

One JSON object per line on **stderr**, always carrying the observability fields:

```json
{"ts":"2026-07-12T10:00:00+00:00","level":"INFO","logger":"agent_kits",
 "message":"pipeline_done","run_id":"run_1f0a…","agent_id":"cloud-runner",
 "model_version":"mock-detector-1.0.0","trace_id":"a1b2c3…","detections":33}
```

Ship these to your log backend (Loki, CloudWatch, Azure Monitor, ELK). Because the
schema is stable you can index on `run_id`, `trace_id`, `agent_id`, `model_version`.

## Required run fields

Every run emits: `run_id`, `agent_id`, `trace_id`, `model_version`, `start_time`,
`end_time` (the last two are in the `RunOutput`; the first four are on every log
line). This is what makes a run auditable end-to-end.

## Health & readiness

- `GET /healthz` — liveness; returns **503** while the pod is draining (SIGTERM) so
  load balancers stop sending new traffic during graceful shutdown.
- `GET /readyz` — readiness for rollout gating.

## Dashboards (recommended panels)

1. **Throughput** — `rate(dvsa_runner_runs_total[5m])`.
2. **Success rate** — success / total (alert < 0.95).
3. **Latency** — avg run latency (alert on p95 breaching SLO).
4. **Saturation** — `dvsa_runner_in_flight` vs `RUNNER_MAX_PARALLELISM` (scale out
   when sustained near the limit).
5. **Retries** — `rate(dvsa_runner_retries_total[5m])` (rising = upstream trouble).

## Deployment recommendations

- Run ≥ 2 replicas behind the Service for availability; scale on `in_flight`
  saturation and CPU.
- Set `terminationGracePeriodSeconds` ≥ your longest expected run so drains finish.
- Scrape `/metrics` every 15–30s; keep 15+ days of history for capacity planning.
- Alert on failure-rate and latency SLOs, not on individual failed runs (retries
  already absorb transient errors).
