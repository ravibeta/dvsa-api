# DVSA API - Drone Video Sensing Analytics API

A professional-grade Django REST API for drone video sensing and analytics, built with modern best practices.

## 🎯 Project Overview

DVSA (Drone Video Sensing Analytics) is a comprehensive API platform for processing, analyzing, and managing drone video data with AI/ML capabilities including computer vision and geospatial analysis.

## 🏗️ Architecture

- **Framework**: Django 5.2.4 with Django REST Framework 3.16.0
- **Database**: PostgreSQL (configurable for SQLite in development)
- **Authentication**: Token-based with JWT support
- **API Standards**: RESTful with proper versioning
- **Async Support**: Celery for background tasks
- **Monitoring**: Comprehensive logging and error tracking

## 📋 Prerequisites

- Python 3.10+
- PostgreSQL 12+
- Redis (for Celery)
- Docker & Docker Compose (optional)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/ravibeta/dvsa-api.git
cd dvsa-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 📤 Direct uploads, analytics jobs and live alerts

Large drone videos never stream through Django. A client asks for a write-only upload URL,
`PUT`s the file straight to cloud storage, then confirms; the confirmation queues an analytics
job whose progress can be polled, pushed over WebSockets, or delivered by signed webhooks.
Storage goes through the cloud-agnostic `core/storage` interface (Azure Blob today, reusing
`AZURE_STORAGE_ACCOUNT` / `AZURE_STORAGE_INPUT_CONTAINER` / `AZURE_ACCOUNT_KEY`). Design notes:
[`docs/batch_jobs_plan.md`](docs/batch_jobs_plan.md).

Every endpoint uses the same JWT as the rest of the API, and everything is owned by the
authenticated user (the UI's `account_id` is the user's id; any other `account_id` gets `403`).

### Running it

```bash
pip install -r requirements/base.txt
python manage.py migrate
redis-server                                   # Celery broker + Channels layer
celery -A config worker -l info                # add --pool=solo on Windows
daphne -p 8000 config.asgi:application         # HTTP + WebSockets (runserver has no WebSockets)
```

`config/settings/development.py` runs Celery tasks eagerly (in the request); set
`CELERY_TASK_ALWAYS_EAGER = False` there to exercise a real worker. New settings are listed
in `.env.example` (`DVSA_UPLOAD_*`, `DVSA_STORAGE_EVENT_TOKEN`, `DVSA_WEBHOOK_*`, `DVSA_ALERT_MIN_VEHICLES`).

### Two-step upload

```bash
API=http://localhost:8000/api/v1
AUTH="Authorization: Bearer $ACCESS"          # from POST $API/users/token/

# 1) Ask for an upload destination (send Idempotency-Key so a retry returns the same asset)
curl -s -X POST $API/media/upload-destination/ -H "$AUTH" -H "Content-Type: application/json" \
     -H "Idempotency-Key: flight-42-cam1" \
     -d '{"filename": "DJI_0042.MP4", "content_type": "video/mp4", "file_size_bytes": 734003200,
          "flight_id": "flight-42", "metadata": {"latitude": 47.61, "longitude": -122.20}}'
# -> {"media_id": "…", "upload_url": "https://<account>.blob.core.windows.net/input/<user>/uploads/flight-42/<media_id>/DJI_0042.mp4?sv=…&sp=cw…",
#     "method": "PUT", "headers": {"x-ms-blob-type": "BlockBlob", "Content-Type": "video/mp4"},
#     "blob_path": "…", "expires_at": "…"}

# 2) Upload straight to storage with the returned URL and headers
curl -X PUT -H "x-ms-blob-type: BlockBlob" -H "Content-Type: video/mp4" \
     --data-binary @DJI_0042.MP4 "$UPLOAD_URL"

# 3) Confirm: verifies the blob exists and its size, then queues a job (default EXTRACT_FRAMES)
curl -s -X POST $API/media/$MEDIA_ID/complete/ -H "$AUTH" -H "Content-Type: application/json" \
     -d '{"job_type": "VEHICLE_TRACKING"}'
# -> 202 {"media_id": "…", "upload_status": "UPLOADED", "job_id": "…", "job": {…}}
```

An expired upload URL can be renewed with `POST $API/media/$MEDIA_ID/upload-destination/`.
Uploading a DJI `.srt` sidecar with the same `flight_id` captures its GPS track so vehicle
detections are placed at the drone's position. Instead of step 3, an Azure Event Grid
`BlobCreated` subscription can call `POST $API/media/events/azure/?token=$DVSA_STORAGE_EVENT_TOKEN`
(the validation handshake is answered automatically).

### Jobs

| Job type | Runs | Output |
|---|---|---|
| `EXTRACT_FRAMES` | the `/videos/extract-frames/` service | JPEG frames beside the upload |
| `VEHICLE_TRACKING` | the `/videos/vehicles/` pipeline | annotated MP4, JSON report, GeoJSON detections |
| `GEOSPATIAL_MAPPING` | the `/videos/orthomosaic/` job (needs Docker + ODM) | orthomosaic JPEG |
| `FULL_PIPELINE` | not a bulk job: the full video indexer workflow is requested per video | `400` |

```bash
curl -s -X POST $API/jobs/ -H "$AUTH" -H "Content-Type: application/json" \
     -d '{"media_id": "'$MEDIA_ID'", "job_type": "GEOSPATIAL_MAPPING"}'      # 202
curl -s $API/jobs/$JOB_ID/ -H "$AUTH"         # status, progress_percent, result_summary, download_urls
curl -s "$API/jobs/?status=COMPLETED&updated_after=2026-11-27T00:00:00Z" -H "$AUTH"   # incremental sync
curl -s -X POST $API/jobs/$JOB_ID/cancel/ -H "$AUTH"   # cooperative: stops at the next stage boundary
curl -s -X POST $API/jobs/$JOB_ID/retry/ -H "$AUTH"    # FAILED/CANCELLED -> new attempt, same job id
```

### Webhooks

```bash
curl -s -X POST $API/webhooks/ -H "$AUTH" -H "Content-Type: application/json" \
     -d '{"target_url": "https://example.com/dvsa-hook", "events": ["job.completed", "alert.detection_triggered"]}'
# -> 201 {..., "secret_key": "…"}   (shown only once; use "*" for every event)
curl -s -X POST $API/webhooks/$HOOK_ID/test/ -H "$AUTH"        # signed "ping"
curl -s $API/webhooks/$HOOK_ID/deliveries/ -H "$AUTH"          # delivery log
```

Events: `media.uploaded`, `job.completed`, `job.failed`, `job.cancelled`,
`alert.detection_triggered`, `ping`. Targets must be public `https` URLs (development also
allows `localhost`). Failed deliveries (5xx, 408/429, network errors) are retried 3 times
with 10 s / 20 s / 40 s backoff; the delivery id stays the same so receivers can de-duplicate.

```http
POST /dvsa-hook HTTP/1.1
Content-Type: application/json
X-DVSA-Event: alert.detection_triggered
X-DVSA-Delivery-ID: 3f0c…
X-DVSA-Signature: t=1795800000,v1=5d41402abc4b2a76b9719d911017c592…

{"created_at":"2026-11-27T22:14:03+00:00","data":{"confidence":0.81,"count":2,"frame_timestamp":0.5,
 "geojson":{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point",
 "coordinates":[-122.2,47.61]},"properties":{"track_id":1,"class":"car","state":"MOVING"}}]},
 "job_id":"…","media_id":"…","flight_id":"flight-42"},"id":"…","type":"alert.detection_triggered"}
```

Verify with HMAC-SHA256 over `f"{t}.{raw_body}"`:

```python
from apps.notifications.signing import verify_signature   # or the equivalent few lines:
# import hmac, hashlib, time
# t, v1 = (p.split("=", 1)[1] for p in header.split(","))
# ok = hmac.compare_digest(v1, hmac.new(secret.encode(), f"{t}.".encode() + raw_body, hashlib.sha256).hexdigest()) \
#      and abs(time.time() - int(t)) < 300
assert verify_signature(secret, raw_body, request.headers["X-DVSA-Signature"])
```

### WebSockets

Authenticate with `?token=<access JWT>` or the subprotocol pair `bearer, <access JWT>`;
unauthenticated sockets are closed with code `4401`.

```javascript
const alerts = new WebSocket(`ws://localhost:8000/ws/alerts/?token=${access}`);
alerts.onmessage = (e) => console.log(JSON.parse(e.data));   // same envelope as the webhook body

const live = new WebSocket("ws://localhost:8000/ws/flights/flight-42/telemetry/", ["bearer", access]);
live.onopen = () => live.send(JSON.stringify({type: "telemetry", lat: 47.61, lon: -122.2, alt: 80, t: 12.5}));
live.onmessage = (e) => console.log(JSON.parse(e.data));     // {type, flight_id, data} for the owner's sockets
```