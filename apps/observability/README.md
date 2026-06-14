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
| `sinks.py` | core | lazy | `CommentarySink` ABC, `Null`/`InMemory`/`DjangoModel`/`OTel`/`Tee` sinks, `get_sink()` |
| `aggregation.py` | core | no | `aggregate_events()` — query-time roll-ups over raw events |
| `otel.py` | core | no | OTLP/HTTP+JSON payload builders (logs/metrics/traces) + `OTLPHttpExporter` |
| `llm.py` | core | lazy | `LLMClient` interface, `Echo` (offline) + `AzureOpenAIChat` clients |
| `vlm.py` | core | no | `AzureVLMCommentator` — model-backed commentary behind the `Commentator` interface |
| `agents.py` | core | no | `SemanticAggregatorAgent` — rolls low-level events into higher-level scene commentary |
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
| `POST` | `agents/summarize/` | Run the semantic agent over stored events (`{"video_id"}` and/or `{"trace_id"}`, optional `scope`) → higher-level event |

## Enabling

Off by default — existing vision runs are unaffected. Set:

```bash
COMMENTARY_ENABLED=True      # turn on the task emit hook
COMMENTARY_SINK=db           # null | db | memory | otel ; comma-list fans out, e.g. "db,otel"
COMMENTARY_COMMENTATOR=template
```

When enabled, `run_video_analysis` emits commentary after each run via a single
guarded hook (failures are logged, never propagated — the vision run always wins).

## MELT / OpenTelemetry export (Phase 3)

Commentary maps onto the three OTel signals and ships over **OTLP/HTTP+JSON**
(stdlib only, no SDK, no extra deps) to any collector:

* **Logs**    ← `commentary` text (one log record per event)
* **Metrics** ← each numeric entry in `metrics` → gauge `dvsa.commentary.<name>`
* **Traces**  ← each event → span (`trace_id`/`span_id`/`parent_span_id`)

```bash
COMMENTARY_SINK=db,otel                         # persist and export
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   # collector base (no signal suffix)
OTEL_SERVICE_NAME=dvsa-api
COMMENTARY_OTEL_SIGNALS=logs,metrics,traces     # subset allowed
```

The `to_otlp_*` builders in `otel.py` are pure and unit-tested; `OTLPHttpExporter`
is best-effort (transport errors are swallowed) so export never breaks a run.

## Migrations

Consistent with the rest of the repo, no migration is committed (the
`migrations/` package is present and ready). Generate at deploy:

```bash
python manage.py makemigrations observability && python manage.py migrate
```

## Agentic / model-backed commentary (Phase 4)

Two model-backed layers, both with deterministic offline fallbacks so they never
break a run and stay testable without credentials:

* **`AzureVLMCommentator`** (`COMMENTARY_COMMENTATOR=vlm`) — generates richer
  per-frame commentary via an `LLMClient`; on any model error it falls back to
  the template commentary.
* **`SemanticAggregatorAgent`** (`POST agents/summarize/`) — consumes low-level
  per-frame events and emits a higher-level scene/trace summary, sharing the
  source `trace_id` and linking the children's span ids (`derived_from_spans`),
  correlated by `correlation_key`.

The LLM backend is provider-agnostic behind `LLMClient`; the shipped
implementation targets the repo's existing **Azure OpenAI** integration over its
REST API (stdlib only, no new dependency):

```bash
COMMENTARY_COMMENTATOR=vlm
COMMENTARY_LLM=azure            # echo (offline default) | azure
AZURE_OPENAI_ENDPOINT=...       # existing settings reused
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_GPT_DEPLOYMENT=...
AZURE_OPENAI_API_VERSION=2024-06-01
```

## Roadmap

* **Phase 3 — MELT/OTel:** ✅ done.
* **Phase 4 — Agentic commentary:** ✅ done.
* **Next:** thread raw frames into `AzureVLMCommentator` for true VLM (image)
  prompts; add an Anthropic `LLMClient` implementation; scheduled trace-level
  summarisation.
