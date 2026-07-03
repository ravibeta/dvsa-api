# `urban_accident_example` — deterministic accident detector

A reference reasoning model that flags **accidents on urban streets** from drone
video *metadata* (object tracks). It is pure Python (no ML dependencies), fully
deterministic, and offline — ideal as a template for real detectors and as a CI
smoke test of the pluggable reasoning pipeline.

## How it decides

It reports an `accident` when both are true:

1. **Convergence** — two tracks come within `proximity_threshold` px of each
   other within `time_window_s` seconds; and
2. **Sudden deceleration** — one of them shows a speed drop between consecutive
   samples exceeding `deceleration_threshold`.

Otherwise it reports `nominal`.

## Input / output

See `sample_input.json` and `sample_output.json`. Each track carries `samples`
(`t`, `x`, `y`, optionally `vx`/`vy`). Run it via the API once the server is up:

```bash
curl -X POST http://localhost:8000/api/reasoning/urban_accident_example/infer \
     -H 'Content-Type: application/json' \
     --data @models/reasoning/urban_accident_example/sample_input.json
```

Expected: an `actions[0]` of `{"type": "anomaly", "label": "accident", "confidence": 0.95, ...}`
plus an explainable `reasoning_trace`.

## Config

Tune thresholds in `manifest.json`'s `config` block (or a `config.json` sidecar):
`proximity_threshold`, `time_window_s`, `deceleration_threshold`.
