# Plan: direct uploads, analytics job queue, live alerts

Source spec: `batch_jobs.md` (workspace root). This plan adapts that spec to the repository as it is
today, using the owner's decisions of 2026-11-27. Where the two disagree, this plan wins.

## 0. Owner decisions

| # | Decision |
|---|---|
| 1 | The **only** owner is the Django `User` who registered through the UI (`POST /api/v1/users/register/`). dvsa-platform sends `account_id = String(user.id)` (`src/lib/auth.tsx:100`). The new endpoints take ownership from `request.user` and **reject** any `account_id` other than `str(request.user.pk)`. There are no aliases and no organizations. |
| 2 | Use the existing **input** container (`AZURE_STORAGE_INPUT_CONTAINER`), not `drone-raw-media`. Object keys and ids must stay unique across retries and recovery (see section 2). |
| 3 | GeoJSON: use the drone's latitude/longitude when the uploaded `metadata` or a `.srt` sidecar provides it; otherwise use pixel coordinates with `"crs": "pixel"`. Only **capture** the uploaded telemetry. Existing pipelines are not georeferenced. |
| 4 | Cancellation is **cooperative**. It follows the existing pattern in `dvsa_api/mcp/executor.py` and `session_manager.py`: a cancel object is checked between stages. The flag is held in the database so it works across Celery processes. |
| 5 | Webhook targets must be HTTPS. Development may allow `http://localhost` through a setting. |
| 6 | Ship everything as **one PR**. |
| 7 | Keep it cloud-agnostic. The new code depends on a small `ObjectStore` interface; only Azure is implemented now, and AWS S3 can be added later without touching these features. **Existing Azure storage code (`core/azure/*`, `apps/videos/*`) is not modified.** It is only imported. |
| 8 | Job types: `VEHICLE_TRACKING` (vehicles), `GEOSPATIAL_MAPPING` (orthomosaic), `EXTRACT_FRAMES` (the `/extract-frames/` service from PR #23; the **default** when an upload completes). `FULL_PIPELINE` is **declared but rejected** with `400`. It is reserved for the full video indexer run followed by its frame extractions, object detections and other indexer stages. Because that is costly, it is never a bulk job and must be requested one video at a time. There is no `ANOMALY_SCAN`. |
| 9 | Bulk jobs **never create a `VideoEntity`**: its `post_save` runs `ingest_video`, which is the costly indexer run reserved for `FULL_PIPELINE`. Runners use the `MediaAsset` UUID as the video key, so outputs land at keys that can't collide with existing ones. |

## 1. Repository facts the design relies on

- Tenancy is `VideoEntity.account_id`, a string holding the user's pk. There is no `Flight` or organization model, so `flight_id` is a free-text grouping key scoped to the owner.
- Settings live in `config/settings/{base,development,testing,production}.py`. Celery is set up in `config/celery.py`, with eager mode in development and testing. Redis is in `docker-compose.yml`.
- `config/asgi.py` is plain Django ASGI and also calls `local_qwen_server.start_if_enabled()`.
- `REST_FRAMEWORK` uses JWT (simplejwt), `StandardResultsSetPagination` and drf-spectacular.
- Routes: each app owns its `urls.py`, and `config/urls_api.py` includes it under `/api/v1/`.
- Reusable services, imported unchanged:
  - `apps.videos.vehicles.track_vehicles` (sync, never raises)
  - `apps.videos.orthomosaic_jobs.run_job` (sync, never raises)
  - `apps.videos.frame_extract.extract_frames` (the service behind `/extract-frames/`)
  - `apps.videos.vehicles.fresh_source_url` (re-signs a stale SAS)
  - `core.azure.survey_corners.SRT_LAT` / `SRT_LON` (DJI `.srt` parsing)
- A `VideoEntity` `post_save` triggers synchronous indexing (`ingest_video`) and an orthomosaic thread. Bulk jobs therefore **never create one** (decision 9).
- `frame_extract.extract_frames` uses Video Indexer key frames when VI is configured and stride sampling otherwise. `EXTRACT_FRAMES` calls it unchanged.
- The repo has no `db/migrations/`. Each new app gets its own `migrations/` folder, which is allowed.

## 2. Layout

```
core/storage/                      # cloud-agnostic object storage (new; Django-free)
  __init__.py                      # get_object_store() -> chosen by DVSA_OBJECT_STORE ("azure")
  base.py                          # ObjectStore protocol, UploadTarget, ObjectInfo, ObjectCreated
  azure_blob.py                    # AzureBlobObjectStore (uses AzureEnvironmentConfig read-only)
apps/media/                        # Feature 1
  models.py  serializers.py  views.py  urls.py  services.py  admin.py  apps.py  migrations/
apps/jobs/                         # Feature 2
  models.py  serializers.py  views.py  urls.py  runners.py  tasks.py  cancel.py  admin.py  apps.py  migrations/
apps/notifications/                # Feature 3
  models.py  serializers.py  views.py  urls.py  signing.py  tasks.py  broadcast.py
  consumers.py  routing.py  ws_auth.py  url_safety.py  admin.py  apps.py  migrations/
config/asgi.py                     # ProtocolTypeRouter(http, websocket); keeps the qwen startup call
```

### `ObjectStore` interface (`core/storage/base.py`)

```python
@dataclass(frozen=True)
class UploadTarget:  url: str; method: str; headers: dict; expires_at: datetime
@dataclass(frozen=True)
class ObjectInfo:    key: str; size: int; etag: str; content_type: str | None
@dataclass(frozen=True)
class ObjectCreated: key: str; size: int | None; etag: str | None

class ObjectStore(Protocol):
    provider: str                       # "azure" (later "aws")
    bucket: str                         # Azure container / S3 bucket
    def presign_upload(self, key, content_type, ttl: timedelta) -> UploadTarget: ...
    def presign_read(self, key, ttl: timedelta) -> str: ...
    def head(self, key) -> ObjectInfo | None: ...                # None when missing
    def parse_created_events(self, body, headers) -> tuple[list[ObjectCreated], dict | None]: ...
        # returns (events, handshake_response); Azure Event Grid validation handshake here,
        # S3->SNS SubscriptionConfirmation would live in the AWS implementation later
```

On Azure, `presign_upload` returns a blob SAS scoped to that one key, with permissions create and write only (no read or list) and a 30-minute lifetime (`DVSA_UPLOAD_SAS_MINUTES`). It also returns the header `x-ms-blob-type: BlockBlob` and method `PUT`. Models store `storage_provider`, `bucket` and `object_key`, never Azure-specific field names. The API also returns `blob_path`, set equal to `object_key`, because the spec names it.

### Identifiers that are safe to retry

- `MediaAsset.id` is a UUID4. The object key is `{user.pk}/uploads/{flight_id or "_"}/{media_id}/{safe_filename}`. Because the key contains the media id, two uploads of the same file never collide.
- `upload-destination` accepts an optional `Idempotency-Key` header (unique per owner). Repeating a call with the same key returns the same `PENDING` asset with a fresh SAS instead of creating a duplicate.
- `POST /media/{id}/upload-destination/` re-issues an expired SAS for the same `PENDING` asset when the client needs to resume.
- `complete` is idempotent. A second call on an `UPLOADED` asset returns the existing job (`202`, same `job_id`).
- `AnalyticsJob.id` is a UUID4. Each dispatch writes a new `celery_task_id` and increments `attempt`.
- Job outputs are recorded under `result_summary.outputs` with their own keys. Existing pipelines keep their existing blob names.
- Webhook deliveries use a UUID `delivery_id`, sent as `X-DVSA-Delivery-ID`. It stays the same across retries so consumers can de-duplicate.

## 3. Feature 1: two-step direct upload

**`MediaAsset`**
- Fields: `id` (UUID), `owner` (FK `User`), `flight_id` (char, blank), `filename`, `content_type`, `file_size_bytes`, `storage_provider`, `bucket`, `object_key` (unique), `upload_status` (`PENDING|UPLOADED|FAILED`), `etag`, `checksum_sha256` (optional), `metadata` (JSON), `idempotency_key` (blank), `created_at`, `updated_at`, `uploaded_at`.
- Constraint: (`owner`, `idempotency_key`) is unique when the key is non-empty.

**Endpoints** (`IsAuthenticated`, owner-scoped; another user's asset returns 404)

| Method and path | Behaviour |
|---|---|
| `POST /api/v1/media/upload-destination/` | Body: `filename`, `content_type`, `file_size_bytes`, optional `flight_id`, `metadata`, `account_id`.<br>Validation: extension and MIME allowlist (`.mp4 .mov .srt .json`); size at most `DVSA_UPLOAD_MAX_BYTES` (default 20 GB); filename sanitized; `account_id`, if sent, must equal `str(user.pk)` or the call returns 403.<br>Returns `201` with `{media_id, upload_url, method, blob_path, object_key, expires_at, headers}`. |
| `POST /api/v1/media/{id}/upload-destination/` | Re-issues the SAS for a `PENDING` asset. |
| `GET /api/v1/media/` and `/media/{id}/` | List or show the owner's assets. |
| `POST /api/v1/media/{id}/complete/` | Calls `head()`. If the object is missing, returns `409 {"error": "object not found"}` and the asset stays `PENDING`.<br>If the size differs from the declared size, the asset is marked `FAILED` with `422`.<br>Otherwise it records `etag` and size and sets `UPLOADED`. A `.srt` asset has its latitude/longitude samples captured into `metadata.telemetry`.<br>For videos, queues an `AnalyticsJob` (type from the body, default `EXTRACT_FRAMES`; `FULL_PIPELINE` is rejected with 400) and returns `202 {media_id, upload_status, job_id}`. For sidecars it returns `200`.<br>Emits `media.uploaded`. |
| `POST /api/v1/media/events/{provider}/` | `AllowAny`, authenticated by `?token=` compared in constant time with `DVSA_STORAGE_EVENT_TOKEN`; disabled when that is empty.<br>Handles the Event Grid validation handshake. For each `BlobCreated` whose key maps to a `PENDING` asset, runs the same completion logic. Unknown keys are ignored. |

## 4. Feature 2: analytics job queue

**`AnalyticsJob`**
- Fields: `id` (UUID), `owner` (FK `User`), `media_asset` (FK), `celery_task_id`, `attempt`, `job_type` (`VEHICLE_TRACKING|GEOSPATIAL_MAPPING|EXTRACT_FRAMES|FULL_PIPELINE`; the last is declared only), `model_name`, `model_version`, `parameters` (JSON), `status` (`QUEUED|PROCESSING|COMPLETED|FAILED|CANCELLED`), `progress_percent` (0–100), `stage`, `result_summary` (JSON), `error_message`, `created_at`, `updated_at` (indexed), `started_at`, `completed_at`.
- Status changes go through `transition(job, new, **fields)`, which allows only legal edges (`QUEUED→PROCESSING|CANCELLED`, `PROCESSING→COMPLETED|FAILED|CANCELLED`, `FAILED|CANCELLED→QUEUED` on retry). The write is a conditional `UPDATE ... WHERE status IN (...)`, so it can't race with a cancel.

**Endpoints**

| Method and path | Behaviour |
|---|---|
| `POST /api/v1/jobs/` | Body: `media_id`, `job_type`, optional `model_name`, `model_version`, `parameters`. The asset must be the owner's and `UPLOADED`. Returns `202`. `FULL_PIPELINE` returns `400 {"error": "FULL_PIPELINE is not supported for bulk jobs; request it per video"}`. |
| `GET /api/v1/jobs/` | Owner's jobs, paginated. Filters: `status`, `job_type`, `media_id`, `updated_after` (ISO 8601). Ordered by `updated_at`, then `id`, for incremental ETL. |
| `GET /api/v1/jobs/{id}/` | Status, progress, stage, timestamps, `result_summary`. |
| `POST /api/v1/jobs/{id}/cancel/` | Sets `CANCELLED` (a conditional update) and calls `revoke(task_id)` without `terminate`, which only drops tasks that haven't started. A running task stops at its next check. Returns `200`; a job already finished returns `409`. |
| `POST /api/v1/jobs/{id}/retry/` | Moves `FAILED` or `CANCELLED` back to `QUEUED`, increments `attempt` and dispatches a new task. This is the recovery path. |

**Cancellation** (`apps/jobs/cancel.py`): `JobCancelToken(job_id)` exposes `is_set()`, which reads `status == CANCELLED` from the job row on every check (checks only happen at stage boundaries), so it has the same interface as the `threading.Event` the MCP executor checks. The runner checks it before every stage and inside progress callbacks, then raises `JobCancelled`. The task catches that and leaves the job `CANCELLED`.

**Task** (`apps/jobs/tasks.py`): `process_drone_video_job(job_id)`, using `@shared_task(bind=True, acks_late=True)`.

1. Move to `PROCESSING` (conditional). If that fails, the job was cancelled or another worker owns it, so the task exits.
2. Run the stages from `runners.py`. Each one reports progress into its own slice of 0–100.
   - Every stage gets `account_id=str(owner.pk)`, `video_key=str(media.id)`, and a read SAS for the asset from `presign_read`.
   - `VEHICLE_TRACKING`: `track_vehicles(account_id, video_key, read_url, ...)`. Outputs go to `{pk}/images/{media_id}/vehicles/`. If it returns `BUSY`, the job moves back to `QUEUED` with `stage="waiting for slot"` and is re-dispatched after 60 s, without counting a new attempt. It fails after 10 waits.
   - `GEOSPATIAL_MAPPING`: `orthomosaic_jobs.run_job(account_id, video_key, read_url)`. Output goes to `{pk}/images/{media_id}/orthomosaic.jpg`.
   - `EXTRACT_FRAMES`: `frame_extract.extract_frames(cfg, read_url, account_id=..., video_id=None, stride_seconds=parameters.stride or 10)`. Frames go beside the upload at `{pk}/uploads/.../{media_id}/images/frameN.jpg`. No `ImageEntity` rows are created, because there is no `VideoEntity`. `result_summary` records the frame source, count and blob names.
   - `FULL_PIPELINE`: never dispatched. It is rejected at the API.
3. For `VEHICLE_TRACKING`, build the GeoJSON `FeatureCollection` described below. Then set `COMPLETED` with `result_summary = {outputs, summary, geojson}`, then call `broadcast_live_alert(owner, "job.completed", ...)`.
   - If vehicles are detected and the count reaches `DVSA_ALERT_MIN_VEHICLES` (default 1), it also sends `alert.detection_triggered`.
4. On any other exception: set `FAILED` with the error truncated to 500 characters, and send `job.failed`. The task never raises into Celery except through the explicit `retry`.

**GeoJSON**: there is one feature per confirmed track, taken from `vehicles_tracking.json`.
- If the asset's `metadata` has `latitude`/`longitude`, or a `.srt` sidecar with the same `flight_id` gave telemetry, the feature is a `Point` at the drone position nearest the track's first sighting.
- Otherwise it is a `Point` in reference-plane pixels, and the collection carries `"crs": {"type": "name", "properties": {"name": "pixel"}}`.
- Each feature's properties are `track_id`, `class`, `state`, `confidence`, `first_t` and `last_t`.

**Worker setup**: `docker-compose` already runs `celery -A config worker`. On Windows use `--pool=solo`. Cancellation doesn't depend on the pool, because it's cooperative.

## 5. Feature 3: webhooks and WebSockets

**Models**
- `WebhookSubscription`: `id` (UUID), `owner`, `target_url`, `secret_key` (`secrets.token_urlsafe(32)`, returned **only** in the create response), `events` (JSON list, validated against `EVENT_TYPES`; `["*"]` means all), `is_active`, `created_at`, `updated_at`.
- `WebhookDeliveryLog`: `id` (UUID, also the delivery id), `subscription`, `event_type`, `payload`, `response_status_code`, `attempt_count`, `delivered_at`, `error`, `created_at`.

Event types: `media.uploaded`, `job.completed`, `job.failed`, `job.cancelled`, `alert.detection_triggered` and `ping`.

**Endpoints**

| Path | Behaviour |
|---|---|
| `/api/v1/webhooks/` | Owner-scoped `GET` list and `POST` create. |
| `/api/v1/webhooks/{id}/` | `GET`, `PATCH`, `DELETE`. |
| `/api/v1/webhooks/{id}/test/` | Queues a `ping`. Returns `202` with the `delivery_id`. |
| `/api/v1/webhooks/{id}/deliveries/` | Delivery log, paginated. |

**Target URL check** (`url_safety.py`, run on create, on update, and again just before each send):
- The scheme must be `https`. `http` is allowed only when `DVSA_WEBHOOK_ALLOW_INSECURE_LOCALHOST=true` (on by default in development) and the host is localhost.
- Private, loopback, link-local, reserved and multicast addresses are rejected after DNS resolution.
- Redirects are not followed.

**Signing** (`signing.py`, Stripe-style):

```
X-DVSA-Event: job.completed
X-DVSA-Delivery-ID: <uuid>
X-DVSA-Timestamp: <unix seconds>
X-DVSA-Signature: t=<ts>,v1=<hex HMAC-SHA256(secret, f"{ts}.{raw_body}")>
```

`verify_signature(secret, body, header, tolerance_s=300)` is shipped too, so consumers and tests can verify deliveries.

**Delivery** (`tasks.deliver_webhook(delivery_id)`):
- Calls `requests.post` with a 10-second timeout and `allow_redirects=False`.
- Any 2xx counts as delivered. Retries use `autoretry_for`, `retry_backoff=True` and `max_retries=3`.
- `attempt_count` and `error` are updated on each attempt.

**Broadcast** (`broadcast.py`): `broadcast_live_alert(owner_id, event_type, payload)`
1. `async_to_sync(channel_layer.group_send)(f"user-{owner_id}", {...})`. If no layer is configured, it is skipped with a logged warning.
2. For each active subscription of that owner matching `event_type` (or `*`), it creates a `WebhookDeliveryLog` and calls `deliver_webhook.delay(id)` from `transaction.on_commit`.

The envelope is `{id, type, created_at, data}`. `data` for `alert.detection_triggered` is `{job_id, media_id, flight_id, geojson, confidence, frame_timestamp}`.

**WebSockets** (`channels`, `channels-redis`, `daphne`)
- `CHANNEL_LAYERS` uses `channels_redis` on `REDIS_URL` or `CELERY_BROKER_URL`. Testing uses `InMemoryChannelLayer`.
- `config/asgi.py` becomes `ProtocolTypeRouter({"http": django_asgi_app, "websocket": JWTAuthMiddleware(URLRouter(...))})` and keeps the existing qwen startup call.
- `ws_auth.JWTAuthMiddleware` accepts a token from `?token=<access>` or from `Sec-WebSocket-Protocol: bearer, <access>`, validated with simplejwt `AccessToken`. Without a valid token the connection is closed with code 4401.

| Socket | Behaviour |
|---|---|
| `ws/alerts/` | Joins group `user-{pk}` and forwards events. The client may send `{"type": "ping"}`. |
| `ws/flights/{flight_id}/telemetry/` | Joins `flight-{pk}-{flight_id}`, so only the owner's own sockets share the group. Telemetry messages such as `{"type": "telemetry", "lat": ..., "lon": ..., "alt": ..., "t": ...}` are validated and re-broadcast to that group. Inference markers use `{"type": "marker", ...}`. |

**Dev server**: `runserver` is left as it is. WebSockets are served with `daphne config.asgi:application`, which is documented in the README. `daphne` is **not** added to `INSTALLED_APPS`, so today's `runserver` behaviour is unchanged.

## 6. Settings, environment and dependencies

- **`base.py`:**
  - Add apps `apps.media`, `apps.jobs`, `apps.notifications` and `channels`.
  - Add `CHANNEL_LAYERS`.
  - Add the variables `DVSA_OBJECT_STORE="azure"`, `DVSA_UPLOAD_SAS_MINUTES=30`, `DVSA_UPLOAD_MAX_BYTES`, `DVSA_STORAGE_EVENT_TOKEN=""`, `DVSA_WEBHOOK_ALLOW_INSECURE_LOCALHOST=false`, `DVSA_WEBHOOK_TIMEOUT_S=10` and `DVSA_ALERT_MIN_VEHICLES=1`.
- **`development.py`:** set `DVSA_WEBHOOK_ALLOW_INSECURE_LOCALHOST=true`.
- **`testing.py`:** use `InMemoryChannelLayer`.
- **`requirements/base.txt`:** add `channels>=4.1,<5`, `channels-redis>=4.2,<5`, `daphne>=4.1,<5`, and `requests` if it is missing.
- **`.env.example`:** document the new variables. The existing Azure variables are reused, not duplicated.

## 7. Tests (`tests/test_media_upload.py`, `test_analytics_jobs.py`, `test_webhooks.py`, `test_websockets.py`)

The tests use an in-memory `FakeObjectStore` that implements the protocol, plus focused tests of `AzureBlobObjectStore` with `azure.storage.blob` mocked. No network or Azure access is needed.

- **Upload:**
  - The SAS is limited to create and write, scoped to one key, with the right lifetime and headers.
  - Allowlist rejections, and a 403 when `account_id` belongs to someone else.
  - Retrying with the same idempotency key returns the same asset.
  - Re-issuing the SAS works.
  - `complete` returns 409 when the object is missing, 422 when the size differs, and 202 with a job otherwise; a second call returns the same job.
  - `.srt` telemetry is captured.
  - Event Grid: handshake, bad token, and a `BlobCreated` event that completes the asset.
  - Another user's asset returns 404.
- **Jobs:**
  - Legal and illegal transitions, and the conditional update racing a cancel.
  - The eager task runs each job type with mocked runners: progress increases monotonically, and the job reaches `COMPLETED` with GeoJSON in both drone and pixel coordinates.
  - A failure sets `FAILED` and sends `job.failed`.
  - A cancel between stages leads to `CANCELLED`, and the later stages never run.
  - A `BUSY` result triggers a retry.
  - The list filters, including `updated_after`, work.
  - Retry increments `attempt`.
  - Owner isolation.
- **Webhooks:**
  - The signature verifies, and tampering or an expired timestamp fails.
  - The event filter and `*` work.
  - Retries with backoff happen on 5xx and timeouts and stop after 3, with `attempt_count` updated.
  - URL safety: HTTP rejected, private IPs rejected, the localhost dev flag allowed.
  - The secret appears only in the create response.
  - `/test/` sends a ping.
- **WebSockets** (`WebsocketCommunicator`, in-memory layer):
  - A connection without a token, or with a bad one, is closed with 4401.
  - Both the query-string and subprotocol tokens work.
  - `broadcast_live_alert` reaches the owner's socket and not another user's.
  - Telemetry is re-broadcast within the flight group and is isolated per owner.

The full suite must stay green, and no existing test may change.

## 8. Documentation

- **`README.md`:** setup (Redis, worker, daphne), a curl walk-through of the two-step upload (`upload-destination`, then `curl -X PUT -H "x-ms-blob-type: BlockBlob" --data-binary @video.mp4 "$UPLOAD_URL"`, then `complete`, then polling the job), the webhook payload with verification code, and WebSocket connection examples.
- **`.env.example`:** the new variables.

## 9. As built (2026-11-27)

- Validation errors come back as field errors (`serializer.errors`), as `FrameExtractAPIView` does, because `core.exceptions.exception_handler` collapses raised validation errors into "An error occurred".
- Job outputs are stored as object keys. `GET /jobs/{id}/` mints fresh read URLs (`download_urls`, at most 200 frames).
- Webhook targets are re-checked against the SSRF rules just before each send. 4xx responses other than 408, 425 and 429 are not retried.
- `EXTRACT_FRAMES` calls `frame_extract.extract_frames` unchanged, so when Video Indexer is configured it samples VI key frames, as `/extract-frames/` does.

## 10. Out of scope

- An AWS `ObjectStore` implementation.
- Refactoring the existing Azure storage code.
- Georeferencing detections beyond the drone's position.
- UI changes in dvsa-platform.
- The to-dos recorded earlier: the Terraform CI fix and the stale-SAS re-sign for corners and orthomosaic.
