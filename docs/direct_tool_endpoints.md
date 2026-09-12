# Direct tool endpoints

Some of the routines the agentic framework uses internally are also useful on
their own. **Direct tool endpoints** expose one such routine each as a plain
Django/DRF endpoint that runs the tool and returns its result directly —
**bypassing the query, RAG, and agentic-synthesis path** used by
`PUT /api/v1/videos/videos/chat/`. They are for callers who want one specific,
deterministic result rather than a synthesized narrative answer.

## `/corners` — survey-area corner frames

```
PUT /api/v1/videos/corners/        (IsAuthenticated)
form-data / JSON:
    account_id   (required)  the account whose video to analyse
    video_id     (optional)  a VideoEntity pk to pick a specific video
    sas_url      (optional)  analyse this blob directly instead of a VideoEntity
```

Given an account's uploaded drone video, the endpoint runs the corner-extraction
routine (`core/azure/survey_corners.py`, ported from the `survey_corners.py`
reference script) and returns the four **survey-area corner frames**, ordered
bottom-left → top-left → top-right → bottom-right.

It works in two modes, unchanged from the script:

- **telemetry** — if a DJI-style `.SRT` sidecar with `[latitude]`/`[longitude]`
  sits next to the video, the flight track comes straight from GPS.
- **visual** — otherwise the track is estimated by accumulating frame-to-frame
  similarity transforms (rough visual odometry). Here "bottom-left" is relative
  to the drone's initial heading, not true north.

The track is reduced to its minimum-area rotated rectangle and the frame nearest
each rectangle corner is selected.

### Source video

The source video is resolved in this order:

1. an explicit `sas_url`, if given;
2. otherwise the `VideoEntity` named by `video_id` (scoped to `account_id`);
3. otherwise the account's **most recent** `VideoEntity`.

If none can be resolved the endpoint returns `404`.

### Output

Each corner frame is written back to blob storage next to the video's extracted
frames, **overwriting** any previous run:

```
{input_container}/{account_id}/images/{video_pk}/corners/{label}.jpg
```

`video_pk` is the resolved `VideoEntity` pk, or `0` when the video was supplied
by `sas_url` (no entity). The response returns a downloadable, read-only SAS URL
(1-hour expiry) for each corner:

```json
{
  "account_id": "7",
  "video_pk": 42,
  "count": 4,
  "corners": [
    {"label": "1-bottom-left",  "downloadUrl": "https://sadronevideo.blob.core.windows.net/input/7/images/42/corners/1-bottom-left.jpg?...",
     "frame": 4065, "t": 135.5, "x": -252.0, "y": -1591.2, "mode": "visual"},
    {"label": "2-top-left",     "downloadUrl": "...", "frame": 2475, "t": 82.5,  "x": -1809.0, "y": 317.5,  "mode": "visual"},
    {"label": "3-top-right",    "downloadUrl": "...", "frame": 1275, "t": 42.5,  "x": -371.8,  "y": 1783.0, "mode": "visual"},
    {"label": "4-bottom-right", "downloadUrl": "...", "frame": 4560, "t": 152.0, "x": 261.0,   "y": -695.0, "mode": "visual"}
  ]
}
```

## `/extract-frames` — salient / time-stride frames

```
POST /api/v1/videos/extract-frames/        (IsAuthenticated)
form-data / JSON:
    account_id      (required)  the account whose video to sample
    video_id        (optional)  a VideoEntity pk (scoped to account_id)
    video_sas_url   (optional)  alias sas_url — sample this blob directly
    stride          (optional)  seconds between samples on the stride fallback (default 10)
    page / page_size (optional) DRF pagination (body or query params; page_size max 100)
```

Given an account's video, the endpoint extracts JPEG frames and writes them into
the **same blob layout ingestion uses** so nothing else has to change:

```
{account_id}/images/{video_id}/frame{N}.jpg     # N contiguous from 0
```

`video_id` is the `VideoEntity` pk when one is resolved (same resolution order
as `/corners`: `video_sas_url`/`sas_url` → `video_id` → the account's most recent
`VideoEntity`; `404` if none), and is omitted from the path when only a raw SAS
URL is given. Numbering is always contiguous `frame0, frame1, …` — the source
timestamp is returned as `t` and stored on `ImageEntity`, never encoded in the
blob name (`get_uploaded_frames` walks `0,1,2,…` and stops at the first gap).

**Frame source** (top-level `source` field), chosen in order:

1. `existing` — frames already present under the prefix are reused; the video is
   **not** downloaded or decoded again, and fresh read SAS URLs are minted. This
   is what makes repeat calls idempotent.
2. `video_indexer` — Azure Video Indexer key frames (shots/keyframes from the
   insights), when `VideoIndexerClient` is configured.
3. `stride` — one frame every `stride` seconds (default 10), sampled with OpenCV.

Because the frames land in the shared layout with contiguous numbering, the
ingestion pipeline (`SessionAzureEnvironment.ingest_video`) sees them via
`get_uploaded_frames` and **skips** its own every-frame dump — Video Indexer and
search indexing are otherwise untouched.

**Response** — the `StandardResultsSetPagination` envelope plus `account_id`,
`video_pk`, and `source`:

```json
{
  "account_id": "7",
  "video_pk": 42,
  "source": "stride",
  "count": 18,
  "next": "http://.../extract-frames/?page=2&page_size=10",
  "previous": null,
  "results": [
    {"frame_number": 0, "blob_name": "7/images/42/frame0.jpg",
     "sas_url": "https://sadronevideo.blob.core.windows.net/input/7/images/42/frame0.jpg?sv=...", "t": 0.0},
    {"frame_number": 1, "blob_name": "7/images/42/frame1.jpg", "sas_url": "...", "t": 10.0}
  ]
}
```

SAS TTL is 1 hour (as `put_blob_and_sas`). When a `VideoEntity` exists, one
`ImageEntity` per frame is persisted (`sas_url`, `timestamp`).

## Testing

`tests/test_corners.py` runs fully offline. The pure geometry (corner ordering
and rectangle sampling) is checked on synthetic tracks, and the endpoint tests
patch the three seams the view uses — the blob download
(`core.azure.blob.download_blob_to_temp`), the extraction
(`core.azure.survey_corners.extract_corner_frames`), and the upload/SAS mint
(`core.azure.blob.put_blob_and_sas`) — so no video runtime, Azure account, or
credentials are needed in CI.

`tests/test_frame_extraction.py` follows the same offline pattern for
`/extract-frames/`: it patches `get_uploaded_frames`, `download_blob_to_temp`,
the samplers (`apps.videos.frame_extract._stride_samples` /
`_video_indexer_samples`), and `put_blob_and_sas` / `read_sas_for_blob`, and
also asserts the ingestion skip (a mocked `ingest_video` does not re-extract when
frames already exist).
