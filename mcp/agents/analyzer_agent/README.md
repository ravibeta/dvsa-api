# `analyzer_agent`

Reasoning-enabled analyzer (capability `analyzer`). Pulls the scout's evidence
from `context.upstream` and confirms anomalies by calling a DVSA reasoning model
through `dvsa_api.reasoning.call_model` (default `urban_accident_example`,
configurable via `config.reasoning_model`). Offline-safe: the reasoning model
returns a deterministic synthetic result, and a local heuristic is used if the
reasoning layer is unavailable. No extra dependencies.
