# `_template_model` — reasoning adapter template

Copy this folder to `models/reasoning/<your_model_name>/` and edit the three
files to register a new reasoning model. **No other code changes are required** —
the model is auto-discovered at startup (or on first request) and becomes
callable at `POST /api/reasoning/<your_model_name>/infer`.

## Files

| File | Purpose |
| --- | --- |
| `manifest.json` | Declares `name`, `version`, `type`, `entrypoint`, `capabilities`, and optional `config`. |
| `adapter.py` | Exports a class named `ReasoningModelAdapter` implementing `predict` + `health_check`. |
| `README.md` | Any extra install/run notes (e.g. `pip install ...`) if your model needs deps. |

## Quick start

```bash
cp -r models/reasoning/_template_model models/reasoning/my_model
# edit manifest.json -> "name": "my_model", and implement adapter.py
# then:
curl -X POST http://localhost:8000/api/reasoning/my_model/infer \
     -H 'Content-Type: application/json' \
     -d '{"query": "anything", "tracks": []}'
```

## Contract

`predict(context)` returns:

```json
{
  "actions": [{"type": "label", "label": "nominal", "confidence": 0.5}],
  "reasoning_trace": ["...optional explainability steps..."],
  "metadata": {"model": "my_model", "version": "0.1.0", "tokens": null, "latency_ms": 3}
}
```

`health_check()` returns `{"status": "ok"}` or `{"status": "error", ...}`.

## Config

`manifest.json`'s `config` object (and an optional `config.json` sidecar) is
passed to your adapter as the `config=` keyword. If your adapter takes no
arguments, that also works.

## Remote / Azure models

Set `"type": "remote"` and `"provider": "azure"` in the manifest to route through
the built-in Azure adapter (reads `AZURE_REASONING_ENDPOINT` / `AZURE_REASONING_KEY`
/ `AZURE_REASONING_DEFAULT`). Provide your own `adapter.py` to wrap a different
remote endpoint. **Never commit secrets** — configure them via environment
variables only.
