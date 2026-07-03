# Pluggable reasoning models

Add a custom **reasoning model** to dvsa-api by dropping a folder into
`custom_models/reasoning/<name>/`. The model is auto-discovered and exposed at
`POST /api/reasoning/<name>/infer` with **no other code changes**. This layers on
the Azure Foundry session provider from
[`docs/azure_foundry_integration.md`](azure_foundry_integration.md); both share
the same adapter contract and registry.

## Quick start

```bash
# 1. Copy the template.
cp -r custom_models/reasoning/_template_model custom_models/reasoning/my_model

# 2. Set "name": "my_model" in manifest.json and implement predict() in adapter.py.

# 3. Restart the server (or just call it — discovery also runs lazily).
curl -X POST http://localhost:8000/api/reasoning/my_model/infer \
     -H 'Content-Type: application/json' \
     -d '{"query": "detect anomalies", "tracks": []}'
```

## Endpoints

| Method & path | Purpose |
| --- | --- |
| `POST /api/reasoning/<model_name>/infer` | Run a specific model. |
| `POST /api/reasoning/infer` | Auto-select a model via `?policy=` (default `latency_optimized`). |
| `GET  /api/reasoning/models` | List discovered/registered models + manifests. |

The request body is the `context`; the response is the adapter's `predict` output
verbatim plus `request_id`, `selected_model`, and `server_latency_ms`.

## Manifest contract (`manifest.json`)

| Key | Required | Notes |
| --- | --- | --- |
| `name` | yes | Unique model name (also the URL segment). |
| `version` | yes | Semver string. |
| `type` | yes | `"local"` or `"remote"`. |
| `entrypoint` | no | Adapter module path; default `adapter.py`. |
| `capabilities` | no | e.g. `["anomaly_detection", "explainable_traces"]`. |
| `provider` | no | `"azure"` routes remote models through the built-in Azure adapter. |
| `cost_hint` / `latency_hint_ms` | no | Drive `cost_optimized` / `latency_optimized` selection. |
| `config` / `config_schema` | no | Model config passed to your adapter as `config=`. |

## Adapter contract (`adapter.py`)

Export a class **named `ReasoningModelAdapter`** implementing:

```python
def predict(self, context: dict) -> dict:
    return {
        "actions": [ {"type": "anomaly", "label": "accident", "confidence": 0.95, "bbox": [x, y, w, h]} ],
        "reasoning_trace": ["...optional explainability..."],
        "metadata": {"model": "my_model", "version": "1.0.0", "tokens": None, "latency_ms": 3},
    }

def health_check(self) -> dict:
    return {"status": "ok"}
```

`context` may contain `frames`, `tracks`, `sensor_meta`, `precomputed_features`,
and `query`; tolerate missing/extra keys. The constructor may accept a `config`
keyword or take no arguments.

## Selection policies

`POST /api/reasoning/infer?policy=<policy>` (or `registry.select_model(policy, context)`):

- `by_name` — uses `context["model"]` / `context["model_name"]` or `AZURE_REASONING_DEFAULT`.
- `cost_optimized` — lowest `cost_hint` (local models are free).
- `latency_optimized` — lowest `latency_hint_ms` (local models are fastest).
- `privacy_first` — prefers `type: local`; never routes off-box if a local model exists.

## Telemetry & safety

- Every call logs `request_id`, `model_name`, `model_version`, `latency_ms`,
  `tokens`, and `reasoning_effort` via `dvsa_api.reasoning.metrics` (console sink
  by default; swap with `metrics.set_sink(...)`).
- Local adapters run under `REASONING_TIMEOUT_MS` (default 15000); an overrun
  returns `model_unavailable`. `MAX_REASONING_LATENCY_MS` flags slow calls in
  telemetry.

## Azure / remote models

Set `"type": "remote"`, `"provider": "azure"` to use the env-configured adapter
(`AZURE_REASONING_ENDPOINT`, `AZURE_REASONING_KEY`, `AZURE_REASONING_DEFAULT`).
With `AZURE_REASONING_FALLBACK=true`, a local model that raises
`ModelUnavailableError` automatically falls back to the Azure adapter. **Never
embed secrets in code or manifests — use environment variables only.**

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `model_unavailable: no reasoning model registered as '<name>'` | Folder missing `manifest.json`, or `name` mismatch. `GET /api/reasoning/models` lists what was found. |
| Model not discovered | Folder name starts with `.`, or malformed `manifest.json` (a bad manifest disables just that model). |
| `<file> does not export a class named 'ReasoningModelAdapter'` | Rename your adapter class to exactly `ReasoningModelAdapter`. |
| Adapter loads but errors at call time | Check `health_check()` and the `reasoning_trace`; ensure `predict` returns the three required keys. |

## Example

`custom_models/reasoning/urban_accident_example/` is a pure-Python, deterministic
accident detector. Feed it `sample_input.json` to get an `accident` anomaly and
an explainable trace — see its `README.md`.
