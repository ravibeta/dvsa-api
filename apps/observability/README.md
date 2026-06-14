# `apps.observability` — Commentary-Driven Observability Layer

A **parallel, orthogonal** telemetry layer that turns drone-video vision output
into wide, query-time-aggregatable *commentary events*. It never replaces or
mutates the YOLO/classical detection workflows in `apps.analytics.routines`; it
reads their JSON envelopes and emits a separate stream of structured events.

Implements Phases 0–2 of the *"From Video to Commentary"* framework.

## Module map

| Module | Layer | Django? | Purpose |
|---|---|---|---|
| `schema.py` | core | no | `CommentaryEvent` wide event, trace/span id helpers, OTel projection |
| `commentator.py` | core | no | `TemplateCommentator` (deterministic) + `events_from_results` bridge |
| `sinks.py` | core | lazy | `CommentarySink` ABC, `Null`/`InMemory`/`DjangoModel` sinks, `get_sink()` |
| `aggregation.py` | core | no | `aggregate_events()` — query-time roll-ups over raw events |
| `emit.py` | glue | yes | `emit_analysis_commentary()` (task hook), `ingest_event()` (API) |
| `models.py` | store | yes | `CommentaryEventRecord` wide table |
| `views.py` / `serializers.py` / `urls.py` | api | yes | ingest / list / aggregate endpoints |

The `core` modules are stdlib-only and unit-tested in `tests/test_observability.py`
without a database (same approach as `tests/test_routines.py`).

## The wide event

One row per frame/segment carrying, side by side: `commentary` (text),
`attributes` (semantic), `metrics` (derived numbers), `metadata` (context) — plus
the correlation triad `trace_id` (per run) / `span_id` (per routine-on-frame) /
`correlation_key` (`video:<id>|frame:<n>`, the join key across detection →
commentary → agent commentary).

## API (`/api/v1/observability/`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `events/` | Inject a custom event (only `commentary` required; rest server-filled) |
| `GET` | `events/` | List raw events; filter by `trace_id`, `video_id`, `source`, `correlation_key`, `frame_index`, … |
| `GET` | `events/aggregate/` | Query-time roll-up: `?group_by=source&metrics=count,sum:count,avg:mean_score` |

## Enabling

Off by default — existing vision runs are unaffected. Set:

```bash
COMMENTARY_ENABLED=True      # turn on the task emit hook
COMMENTARY_SINK=db           # null | db | memory
COMMENTARY_COMMENTATOR=template
```

When enabled, `run_video_analysis` emits commentary after each run via a single
guarded hook (failures are logged, never propagated — the vision run always wins).

## Migrations

Consistent with the rest of the repo, no migration is committed (the
`migrations/` package is present and ready). Generate at deploy:

```bash
python manage.py makemigrations observability && python manage.py migrate
```

## Roadmap

* **Phase 3 — MELT/OTel:** add `OTelSink` exporting metrics/events/logs/traces
  over OTLP using `to_otel_log_record()` as the seam.
* **Phase 4 — Agentic commentary:** add `AzureVLMCommentator` (uses existing
  `AZURE_OPENAI_*` settings) behind the same `Commentator` interface, plus an
  agent that consumes low-level events and emits higher-level semantic
  commentary correlated by `correlation_key`.
