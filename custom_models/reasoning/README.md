# `custom_models/reasoning/` — bring-your-own **reasoning** models

This is a **separate, special case** of `custom_models/`. Where the rest of
`custom_models/` selects and runs **vision / object-detection** models (ONNX /
PyTorch / YOLO, driven by `models_catalog.json` and the `adapters/` package),
this subtree hosts pluggable **reasoning** models for higher-level analytics
(e.g. urban-street accident detection from object tracks).

The two live side by side under `custom_models/` but do **not** share a runtime:

| | Vision custom models (`custom_models/`) | Reasoning models (`custom_models/reasoning/`) |
| --- | --- | --- |
| Interface | `detector.load()/infer(frame)/close()` → `[{label, score, bbox}]` | `ReasoningModelAdapter.predict(context)/health_check()` |
| Discovery | `models_catalog.json` + `discover_model(path)` | folder scan of `custom_models/reasoning/*/manifest.json` |
| Runtime / registry | `custom_models.registry` / `selector` | `dvsa_api.reasoning.registry` |
| API | analytics routines | `POST /api/reasoning/<name>/infer` |

## Adding a reasoning model

Drop a folder here with a `manifest.json` and an `adapter.py` exporting a
`ReasoningModelAdapter` class — it is auto-discovered and callable at
`POST /api/reasoning/<name>/infer` with no other code changes.

- `_template_model/` — copy this to start a new model.
- `urban_accident_example/` — a deterministic, pure-Python reference detector.

See [`docs/reasoning_integration.md`](../../docs/reasoning_integration.md) for the
full manifest/adapter contract, selection policies, telemetry, and Azure/remote
options.
