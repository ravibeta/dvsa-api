# `scout_agent`

Lightweight first-pass scout (capability `scout`). Ingests `payload.tracks` /
`payload.frames`, and when ≥ `min_tracks_for_candidate` (default 2) tracks are
present it emits an `anomaly_candidate` and forwards the raw evidence for the
analyzer. Pure stdlib, deterministic, offline. No extra dependencies.
