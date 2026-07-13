# Changelog — DVSA Agent Kits

All notable changes to `agent_kits/` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-12

### Added
- **Shared foundation** (`agent_kits/common/`):
  - Pydantic schemas `RunInput`, `Detection`, `RunOutput` (+ `TimeWindow`,
    `Provenance`) with programmatic JSON-schema export (`export_schemas`). A
    dependency-free compatibility shim keeps the package importable without
    Pydantic installed.
  - Adapter layer: `VideoFetcher`, `FrameExtractor`, `InferenceAdapter`,
    `StorageAdapter` protocols with offline reference implementations
    (`LocalVideoFetcher`, `SyntheticFrameExtractor`, `MockInferenceAdapter`,
    `LocalFileStorageAdapter`).
  - Deterministic `run_pipeline` (fetch → extract → infer → store).
  - Structured JSON logging emitting `run_id`, `agent_id`, `model_version`,
    `trace_id`.
  - Utilities: deterministic run-id, stable timestamp sort, IoU, confidence-based
    dedupe, deterministic merge, summary.
- **Kit 1 — Local Interactive**: CLI (`fetch-video`, `extract-frames`,
  `run-inference`, `write-output`, `run`) with dry-run and interactive-confirm
  gating, dynamically loaded skills, `agent_profile.yaml` autonomy controls,
  `agent_harness.py`.
- **Kit 2 — Cloud Autonomous**: FastAPI runner (`POST /run`, `GET /metrics`,
  `GET /healthz`) with input validation, retry handling, graceful shutdown and
  Prometheus-compatible counters; `Dockerfile`, K8s deployment, `.env.example`,
  observability guide.
- **Kit 3 — Orchestration**: parent orchestrator (time-window / spatial-tile
  partitioning with configurable overlap), child worker template, deterministic
  `merge_outputs` (IoU overlap removal + confidence conflict resolution +
  canonical timeline), optional `critic_review` (human sampling + secondary-model
  validation).
- **Kit 4 — Integrations & Triggers**: GitHub Actions workflow example, Slack
  slash-command app (manifest + signature-verified trigger), generic
  signature-verified webhook receiver, payload→`RunInput` conversion, run tracking.
- **Test assets**: deterministic synthetic-video generator with CI fallback.
- **CI**: `.github/workflows/ci-agent-kits.yml` — lint, unit tests, Docker build,
  smoke tests, harness + orchestrator exercises; fully offline, < 5 min.
- **Docs**: per-kit READMEs, top-level README, `SECURITY.md`, security snippets.

### Security
- No secrets committed; all configuration via environment variables.
- HMAC signature verification for Slack and webhook triggers.
- Least-privilege storage guidance and PII-redaction notes in `SECURITY.md`.

[0.1.0]: https://github.com/ravibeta/dvsa-api/releases/tag/agent_kits-v0.1.0
