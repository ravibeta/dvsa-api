# Virtual Location & Timestamp Enrichment — Implementation Plan

> **Status:** PLAN ONLY. No code has been changed. Resume by working through §15 (Resume checklist) and the open questions in §14.
> **Source proposal:** `VirtualLocationtimestamp.docx` (repo root, untracked; note the lowercase "t").
> **Workflow (CLAUDE.md):** restate understanding → get confirmation → plan → commit/push/PR in reviewable stages → plan tests after any logic change. Never touch `db/migrations/` unless asked (see §5.4 — new models *will* need migrations, so that needs explicit approval).

> **Revision 4:** added an open-source photogrammetry provider (OpenDroneMap / OpenSfM) that builds a single orthomosaic from frames extracted from multiple drone tours (new §8.3, phase P7, §12 endpoints, §14 Q17–Q22, §15, §16). This makes orthomosaics a DVSA *output* (today there is no orthomosaic code), and gives Path B a concrete reconstruction provider.
> **Revision 3:** added a post-processing geometry & kinematics engine (new §10.1, phase P6, §14 Q13–Q16, §15, §16). Existing state it builds on: `survey_corners.py` (4 corner *frames*, not stored), `geometry.py` (homography), `tracking.py` (pixel-unit SORT trajectories), and an index that is write-once (no merge/update, no back-fill job).
> **Revision 2 (after reviewing `aerial_location/`):** Path C now builds on the reference script `aerial_location/locate_aerial.py` (see §3, §8.2, P4, §14 Q9–Q12, §15, §16). P0 is unchanged.

---

## 1. Intent in one paragraph

DVSA should, **in the background**, give every extracted frame (a) a reliable **video-relative timestamp** and (b) an explicit **spatial result status**, and populate latitude / longitude / altitude **only to the extent the evidence supports it**. The output is an auditable analytical trajectory, never fictitious flight telemetry. Uncertain visual inference must stay labelled as such all the way into the search index and the chat agents' answers.

Acceptance criterion (from the proposal): *every extracted frame receives a reliable video-relative timestamp and an explicit spatial result status.*

---

## 2. Core semantics (decisions the whole design hangs on)

### 2.1 Two different spatial results — never merged

| Concept | Meaning | Fields |
|---|---|---|
| **Virtual camera location** | Estimated lat/lon/alt of the camera | `camera_location` (geo point), `camera_altitude_m`, `camera_altitude_reference` |
| **Virtual scene location** | Geo point of the frame's reference pixel (e.g. image centre projected onto a surface) | `scene_location` |

Rules:
- `virtual_location = virtual_camera_location` (backward-compat alias); **null** when unavailable.
- Never combine scene lat/lon with camera altitude into one point.
- A valid orthophoto/map match gives *scene* position; it does **not** give camera position without projection model + geometry + calibration. Publish scene-only when camera constraints are missing.

### 2.2 Three vertical quantities — never conflated

1. `height_above_reference_plane_m` (perpendicular distance to a fitted plane)
2. `altitude_gain_from_baseline_m` (change in #1 vs the baseline)
3. `camera_altitude_m` + `camera_altitude_reference` (absolute, with named vertical datum)

Only derive #3 from #1/#2 when a validated vertical displacement exists in the same reference system as the initial altitude. A tilted plane's perpendicular distance is **not** geographic vertical displacement. A terrain-elevation service is **not** a drone-altitude service. The supplied LingBot-Map baseline is the **median of initial frames**, not the exact first frame — the contract must say so.

### 2.3 Timestamps

- **Primary:** `video_offset_us` — elapsed presentation time from the first presented frame (integer microseconds).
- **Optional:** `flight_offset_us` (needs a trusted video-start-in-flight offset), `capture_time_utc` (needs a trusted absolute anchor).
- Upload / processing / extraction wall-clock time **never** substitutes for any of these. Video offset 0 ≠ flight offset 0.
- Never infer time from filenames (`frame{N}.jpg`).

### 2.4 Spatial status vocabulary

`measured` · `estimated` · `interpolated` · `relative_only` · `unavailable` · `rejected`
Separate `horizontal_status` and `vertical_status`. `position_method` ∈ {`telemetry`, `telemetry_interpolated`, `geo_aligned_reconstruction`, `orthophoto_registration` (scene only), `relative_reconstruction`, `none`}.

---

## 3. What exists today (grounding)

| Area | Current state | Consequence for this plan |
|---|---|---|
| `ImageEntity` | Has `timestamp` (TimeField) and `location` (CharField) | TimeField can't hold µs offsets > 24 h or a presentation offset reliably → add a dedicated model rather than overloading it; keep these as a back-compat projection |
| `apps/videos/frame_extract.py` + `FrameExtractAPIView._sync_image_entities` | Blob layout `{account}/images/{video}/frame{N}.jpg`, contiguous numbering; `t` seconds only in response / ImageEntity | Extraction must emit a **timing manifest** (source PTS + timebase) — the numbering is *not* a time source; seeked extraction can return a different frame than requested |
| `VideoEntity` post_save → `ingest_video` | Existing event hook | Natural place to emit `VideoReady` |
| `core/azure/document.build_document` | id `{account}-{frame:04d}`; `path` = full SAS URL; does not populate location/timestamp/video_id | **Two index-integrity bugs (§11) must be fixed first** |
| `core/azure/index_schema.py` | Already declares `location`, `timestamp`, `video_id`, `frame_number`, `prev_id`, `next_id`, `version`, `acl_*`, `annotations` | Extend with the new fields in §11; fields were declared but never populated |
| `curated_frame_urls` | Chooses curated frames for chat | Spatial context for chat is read from stored estimates for these frames |
| `core/azure/survey_corners.py` | SRT GPS telemetry / ORB odometry | Seed for the telemetry provider (Path A) |
| `apps/analytics/routines/tracking.py` | SORT + trajectory_reconstruction (PR #24) | Object-track trajectories ≠ camera trajectory; share nothing but naming care |
| Azure Video Indexer client; Perplexity geolocation → `geotags` | Existing geo hints | Possible weak geographic anchors; must carry provenance + low trust |
| `aerial_location/` (`locate_aerial.py`, README, requirements; untracked) | Standalone CLI: matches ONE query image against georeferenced RGB tiles (SuperPoint+LightGlue or SIFT) → MAGSAC homography → image-center lon/lat; strict geometric gates; statuses `geometrically_supported`/`ambiguous`/`abstained`; footprint GeoJSON, overlay, bootstrap p95. Explicitly "image center, not drone GPS". **Never executed end-to-end** (README: only syntax/byte-compile checked) | Reference implementation for Path C / `scene_location`; must be refactored into an importable provider and validated before being trusted (§8.2) |
| Agentic retrieval (`docs/agentic_retrieval.md`) | `synthesize_from_chat_agent` has a **75 s shared budget**, 90 s/run, 6 tool rounds | Enrichment must **not** run inside it (§6) |
| PR #29 (open) | New index schema, agents-as-code, `core.azure.deploy` | Index changes here should land *after* #29 merges or be rebased onto it |

---

## 4. Target architecture

```
apps/geospatial/              # Django: models, DRF views, serializers, admin, tasks wiring
core/geospatial/              # provider-neutral domain logic (no Django, no Azure SDK imports)
  contracts.py                # dataclasses / protocols: FrameTiming, TimelineManifest, SpatialEstimate, ProviderResult
  policy.py                   # account policy modes, thresholds (max gap, max uncertainty), provider allow-list
  orchestrator.py             # enrich_video(): staged, checkpointed, resumable
  timeline.py                 # PTS → video_offset_us normalisation, VFR, segments
  synchronization.py          # telemetry↔video clock alignment
  alignment.py                # local-metric → geographic transform solve + validation
  interpolation.py            # bounded interpolation in metric space
  validation.py               # gap / discontinuity / plane-fit / scale checks
  provenance.py               # evidence manifest, provider+checkpoint versions
  providers/
    telemetry.py              # SRT/CSV/GPX → normalised samples (wraps survey_corners parsing)
    reconstruction.py         # camera pose + scene geometry provider contract (OpenSfM/ODM implementation: §8.3)
    photogrammetry.py         # multi-tour orthomosaic provider contract (NodeODM / OpenSfM): §8.3
    altitude.py               # async altitude-job contract (LingBot-Map wrapper)
    registration.py           # frame→orthophoto keyframe registration (refactored from aerial_location/locate_aerial.py, §8.2)
  publication/
    frame_metadata.py         # estimate → ImageEntity/API projection
    search_documents.py       # estimate → index document fields
    outbox.py                 # durable index-update outbox
```

Reuse, don't reinvent: DVSA commentary events, adapter discovery, retry/fallback and orchestration abstractions (**to verify exact module names when starting Phase 0** — I have not re-read them in this session).

---

## 5. Data model (`apps/geospatial/models.py`)

### 5.1 Models

- **`VideoSpatialContext`** (one current + revisions): `video`, `revision`, anchors (flight-start offset, UTC start, geographic keyframe anchors, calibration, initial absolute altitude + datum **separate from** initial height above reference plane), telemetry artifact refs, `interpolation_policy`, `configuration_hash`, `video_content_hash`.
- **`FrameTiming`**: `image_entity_id`, `extraction_revision`, `source_pts`, `timebase_numerator`, `timebase_denominator`, `video_offset_us`, `segment_id`, `is_vfr`, `flags`. Unique on (`image_entity`, `extraction_revision`).
- **`SpatialAnalysisRun`**: `video`, `idempotency_key` (unique), states `queued → running → waiting_for_provider → partial | completed | failed | cancelled | superseded`, per-stage checkpoint JSON + artifact refs, lease owner/expiry, provider versions, budget counters, `dry_run`.
- **`FrameSpatialEstimate`** — **immutable** per (frame, run/revision): timestamps (`video_offset_us`, `flight_offset_us?`, `capture_time_utc?`), `camera_lat/lon`, `camera_altitude_m`, `camera_altitude_reference`, `scene_lat/lon`, `height_above_reference_plane_m`, `altitude_gain_from_baseline_m`, `horizontal_status`, `vertical_status`, `position_method`, `horizontal_uncertainty_m`, `vertical_uncertainty_m`, `spatial_segment_id`, `geometry_revision`, `evidence_manifest_id`, `reason_codes[]`.
- **`PublishedSpatialRevision`**: current pointer per video → run (pointer swap is the publish act).
- **`IndexOutbox`**: (`document_id`, `payload_hash`, `state`, `attempts`, `next_attempt_at`) for idempotent index upserts.

### 5.2 Idempotency key
`hash(account_id, video_content_hash, extraction_manifest_revision, spatial_context_revision, provider_versions, configuration_hash)`; acquire via lease so duplicate events don't launch concurrent GPU jobs.

### 5.3 Account isolation
Every model carries/derives `account`; every lookup (artifacts, jobs, frames, provider callbacks) filters on it. Mismatched account/video → reject access **and** publication.

### 5.4 Migrations
New tables require migrations in the **app's own** `migrations/`. CLAUDE.md forbids `db/migrations/` unless asked — **confirm with the user** that adding migrations for the new app is allowed before Phase 1.

---

## 6. Execution model (events, jobs, cancellation)

Triggers (events): `VideoReady`, `FramesExtracted`, `FlightAnchorsUpdated`, `CalibrationUpdated`, `SpatialAnalysisCompleted`.

- API returns promptly with a job id: `{"job_id":"spatial-run-42","state":"queued","timestamps_ready":true,"spatial_status":"pending"}`.
- Run in a **separate worker/queue** with its own timeouts and cancellation — **not** in the chat request path and **not** in `FoundryAgents.synthesize_from_chat_agent` (75 s shared budget).
- Each stage checkpoints to DB/blob; **no DB transaction spans external work**. Worker crash → resume from persisted stage artifacts.
- GPU/provider jobs: container- or process-level cancellation + resource limits (abandoning a waiting thread is not cancellation). External HTTP calls get their own timeouts. Budget exhausted → `partial`, resume only under policy.
- A run is **superseded** (or kept only as historical evidence) if the video, anchors, or extraction manifest changed while it ran.

**Worker technology: open question (§14).** Candidates: Celery/RQ if already present in the repo (to check), Django-Q, or a management-command poller over `SpatialAnalysisRun` with leases — the last has no new infra and fits the lease design.

### 6.1 Analysis DAG

```
probe_video → build_timing_manifest →
   { normalize_telemetry ‖ sample_analysis_frames →
        { reconstruct_camera_trajectory, estimate_reference_plane, register_geographic_keyframes } } →
resolve_coordinate_alignment → construct_validated_trajectory →
enrich_extracted_frames → publish_revision
```

- **Two frame sets:** dense *analysis samples* (for reconstruction; run dense inference **once per video/segment**) vs *curated/extracted frames* (what gets published). The altitude script must consume a **canonical sampled-frame manifest** and join results by `frame_id` + `video_offset_us`, **never by row number**.
- Trim/concat boundaries and reconstruction resets = segment boundaries.

---

## 7. Timeline foundation (Phase 1 core)

1. `probe_video`: ffprobe stream timebase, `start_time`, frame count, VFR detection.
2. Extraction emits for each frame: `source_pts`, timebase, and the **actual** decoded frame time (not requested seek time).
3. Normalise: `video_offset_us = (pts − first_presented_pts) × timebase × 1e6`.
4. Handle: non-zero first PTS, VFR, seek returning a different frame, trim/concat segment boundaries.
5. Persist `FrameTiming`; expose `video_offset_us` via the frames API; keep `ImageEntity.timestamp` as a derived, back-compat projection.
6. Touch point: `apps/videos/frame_extract.py` and `FrameExtractAPIView._sync_image_entities` — minimal, surgical additions to emit the manifest (CLAUDE.md: surgical changes).

---

## 8. Spatial evidence — policy-driven paths

Policy hierarchy, but **validate each source** rather than trusting its label.

| Path | When | Output |
|---|---|---|
| **A. Synchronized telemetry** | SRT/CSV/GPX etc. available | Parse position + altitude refs; resolve telemetry↔video clock; account for camera-vs-aircraft offsets; validate gaps/discontinuities; interpolate only inside bounded valid intervals. `position_method=telemetry` / `telemetry_interpolated`. Keep user telemetry distinct from visual estimates |
| **B. Geo-aligned reconstruction** | No/partial telemetry | Reconstruct poses + geometry → associate landmarks → establish metric scale + orientation → solve local→geographic alignment (compute in **local metric coordinates**) → validate against independent landmarks → transform camera centres |
| **C. Frame→orthophoto registration** | Georeferenced imagery available | Scene positions / geographic constraints on B. **Scene-only** unless camera model + geometry + calibration exist. Implemented from `aerial_location/` (§8.2) |
| **D. Relative only** | Reconstruction ok, no geo alignment | Keep local coordinates in artifacts; publish video offsets and relative height with units; geographic camera fields **null**. Never relabel unscaled units as metres |

### 8.2 Registration provider (Path C) — built from `aerial_location/`

**Mapping of script output → DVSA fields**

| Script output | DVSA field |
|---|---|
| `center` (image-centre lon/lat) | `scene_lat/lon` → index `scene_location`. **Never** `camera_location` |
| `status = geometrically_supported` | `horizontal_status = estimated`, `position_method = orthophoto_registration` |
| `status = ambiguous` (competing distant match ≥ 80 % of best score, > 100 m away) | `horizontal_status = unavailable`, reason code `ambiguous_registration` |
| `status = abstained` | `horizontal_status = unavailable`, reason code `registration_abstained` |
| `registration_bootstrap_p95_m` | `horizontal_uncertainty_m`, documented as **fit stability only**, not true accuracy |
| `reference_horizontal_accuracy_m`, imagery source/date, backend, thresholds | `evidence_manifest` (provenance) |
| `footprint.geojson`, `homography.json`, `alignment_overlay.png`, `search_log.json` | stored run artifacts (audit trail, account-scoped blobs) |

**Design decisions**
1. **Unit of work = keyframe**, not every frame. Select a few keyframes per segment (spread across time, high feature richness); results become geographic anchors consumed by Path B's alignment solve (`register_geographic_keyframes` in the DAG) and, on their own, scene-only estimates for those frames.
2. **Refactor, don't shell out.** Move `Matcher`, `test_match`, tile sources (`raster_tiles`, `api_tiles`), `project` into `core/geospatial/providers/registration.py` as importable functions with no `sys.exit`/file dumping; artifacts are written by the orchestrator. Keep `aerial_location/` as a debug CLI (or fold it in once parity tests pass).
3. **Runs in the worker/GPU container**, never the Django web process (torch + LightGlue + rasterio + pyproj + OpenCV). `--backend sift` is the CPU fallback; backend is recorded in provenance.
4. **VLM proposal step (`--propose`) is off by default.** It uploads imagery to an external model, so it is only allowed under `external_visual_analysis` account policy, and if used it goes through the existing Azure Foundry path (not direct OpenAI). Safer candidate sources: user-supplied `spatial-context` anchors, then low-trust hints (Video Indexer, Perplexity `geotags`) as search centres only — never as answers.
5. **Reference imagery is an explicit dependency**: RGB uint8 GeoTIFFs with CRS, or an authorised ArcGIS Image/MapServer. Recall is bounded by supplied coverage; budget caps (`max_tiles`, `max_exports`) are request bounds, not dollar bounds → account-level cost/request policy. Licensing must permit computational matching and retaining derived overlays. Tokens are never logged.
6. **Planar-homography limits** (relief, roof heights, perspective, imagery date/datum) stay in the result contract; thresholds are heuristics, not calibrated probabilities.
7. **Validate before trusting**: the script has never been run end-to-end. Port with synthetic parity tests first (§15), then a real-imagery fixture once supplied.

### 8.3 Photogrammetry provider — one orthomosaic from many tours (OpenDroneMap / OpenSfM)

**Goal:** a fully open-source, Python-driven pipeline that turns frames extracted from *several* videos (tours) of the same area into **one** georeferenced orthomosaic (plus DSM/point cloud/camera poses), runnable on infrastructure DVSA controls. Today nothing in dvsa-api builds an orthomosaic (see the capability review); the aerial_location tool and Path C only *consume* one.

**Engine choice (to verify versions/flags at implementation time)**
- **OpenDroneMap (ODM)** as the end-to-end engine, run as a **separate NodeODM service** (container) and driven from Python via its REST API / the `pyodm` client: create a task with the selected images + options, poll status, download `orthophoto.tif`, `dsm.tif`, `georeferenced_model.laz`, `cameras.json`, `shots.geojson`, `odm_report`, cancel/remove.
- **OpenSfM** (the SfM core ODM uses) for lighter or custom runs: camera poses, sparse reconstruction, GCP alignment, exposed as the `reconstruction` provider (Path B) without producing a full mosaic.
- **Licensing:** ODM/NodeODM are AGPL-3.0; OpenSfM is BSD-2-Clause. Keep ODM as a separate network service (no linking/embedding in dvsa-api) and have legal confirm AGPL obligations for how it is deployed and offered.

**Contract (`core/geospatial/providers/photogrammetry.py`)**
```
submit(project: OrthomosaicProject, inputs: list[FrameInput], options: dict) -> job_id
status(job_id) -> queued|running|completed|failed|cancelled  (+progress, log tail)
results(job_id) -> {orthophoto_cog, dsm, point_cloud, camera_poses, report, crs, gsd_cm, extent}
cancel(job_id)  # container/task-level cancellation, not just abandoning a thread
```
Provider and ODM/OpenSfM versions, options, and image manifest hash are recorded in the evidence manifest.

**Turning video frames into photogrammetry inputs (the hard part)**
1. **Frame selection:** do not feed every frame. Choose frames by overlap/baseline (e.g. sharpness/blur filter, minimum parallax, target forward ~70-80 % / side ~60-70 % overlap), per tour, using `FrameTiming` for ordering. This is a *third* frame set alongside dense analysis samples and curated frames (§6.1).
2. **Metadata injection:** extracted JPEGs carry no EXIF GPS. For each selected frame write EXIF (GPS lat/lon/alt, timestamp, focal length/camera model if known) from synchronized telemetry (Path A, interpolated only within valid intervals). Frames without trustworthy position are included without GPS so ODM can still do SfM, but the result is then **unreferenced** (local coordinates).
3. **Ground control:** optional `gcp_list.txt` from user-supplied anchors or registered keyframes (Path C, `aerial_location`), giving absolute georeferencing and scale when telemetry is missing.
4. **Camera model:** record lens/sensor info; video is rolling-shutter and compressed, so expect lower accuracy than stills; surface this in the quality status. Use ODM camera/lens options where the camera is known.
5. **Multi-tour merging:** all selected frames of all chosen videos go into **one** ODM task so features match across tours. Require a compatible coverage area and a consistent spatial reference (all with telemetry, or shared GCPs). Handle very large sets via ODM's split-merge or by tiling the submission; handle lighting/time-of-day differences with a per-tour quality report rather than silent blending.

**Outputs and how they feed the rest of the plan**
- **Orthomosaic (COG GeoTIFF)** stored under `{account}/orthomosaics/{project}/…`, with extent, CRS, GSD and per-tile source-frame provenance.
- **Camera poses (`shots.geojson`/`cameras.json`/`reconstruction`)** → per-frame **camera location/altitude** estimates (Path B) with uncertainty and the fitted reference plane; replaces the need for a separate plane fit when ODM succeeds. Heights remain "above reference plane" unless a vertical datum is established (§2.2).
- **Orthomosaic as reference** → input to Path C registration for *other* videos of the same area, and to P6: a **SurveyFrame** can be derived from the mosaic extent (its four corners in map coordinates, a validated ground scale), which makes metre-based corner offsets and ground-unit kinematics trustworthy.
- **Index/chat:** `orthomosaic_id`, `ortho_tile_ref`/pixel position per frame (where the frame lies on the mosaic), `reconstruction_status`, `georeferencing_status` (`georeferenced_telemetry|georeferenced_gcp|unreferenced`).

**Rules**
1. Unreferenced or unscaled results are published as local/relative only; never present them as geographic or metric (consistent with Path D).
2. Inputs, selected frames, options and engine versions are immutable per `OrthomosaicRun`; a changed input set creates a new revision (supersede, don't overwrite).
3. Account isolation on every input, artifact and job; external transfer of frames to a hosted ODM requires the `external_visual_analysis` policy; a self-hosted NodeODM does not leave DVSA infrastructure.
4. Long-running, GPU/CPU- and memory-heavy: own queue, resource limits, timeouts, cost caps, cancellation at task/container level; budget exhaustion → `partial`.
5. Dry-run uses a tiny fixture set and never fabricates a georeference.

**Dependencies:** `FrameTiming` (P1), telemetry normalization (P2) for EXIF, registration/GCPs (P4a) for referencing without telemetry, artifact storage + job model from §5/§6.

### 8.1 Altitude provider (asynchronous contract)
The attachment describes a **local LingBot-Map script** (camera geometry → reference plane fit → perpendicular distance), not a hosted API. DVSA defines its own adapter that can wrap a managed GPU worker, a containerised script, or an authorised external service:

```
POST   /v1/altitude-jobs
GET    /v1/altitude-jobs/{job_id}
GET    /v1/altitude-jobs/{job_id}/results
DELETE /v1/altitude-jobs/{job_id}
```
Submission: `video_artifact_id`, `frame_manifest_id`, `initial_height_above_reference_plane_m`, `mode`, `configuration_revision`.
Result per frame: `frame_id`, `video_offset_us`, `height_above_reference_plane_m`, `gain_from_baseline_m`, `unit`, `reference_plane_id`, `segment_id`, `quality{accepted, reason_codes[]}`.
Record provider + checkpoint versions, scale source, plane diagnostics, baseline definition. Poor plane fit ⇒ reject plane-based heights.

---

## 9. Per-frame population (bounded interpolation)

Once the video-wide trajectory exists, evaluate at each extracted frame's **actual** timestamp:
- Interpolate in a metric coordinate representation (not raw longitude across discontinuities).
- Only between accepted samples **in the same segment**; honour max time-gap and uncertainty thresholds.
- Never bridge resets, cuts, or invalid provider intervals; **no extrapolation** before first / after last accepted sample.
- Mark `interpolated` explicitly. Frames outside coverage: timestamp published, spatial fields `unavailable` (reaching video end does not justify a final position).

Orchestrator shape (from the proposal, checkpointed per stage): `acquire_or_resume_run → build_timeline → persist_frame_times → telemetry.normalize → reconstruction.run_or_reuse → altitude.run_or_reuse → resolve_alignment → validate_and_build_trajectory → evaluate_frame × N → publish_revision_and_outbox`.

**Dry-run mode:** deterministic timestamps + clearly marked unavailable spatial fields or fixture results. Must not fabricate geographic coordinates.

---

## 10. Publication (durable, revision-safe)

- Immutable estimates + a **current published-revision pointer** per video.
- Outbox → idempotent index upserts; publication retryable independently of analysis (indexing failure ⇒ retry outbox, never rerun inference).
- Fan-out targets: DB/ImageEntity projection, Azure AI Search, frame APIs, chat grounding, MCP.

### 10.1 Post-processing geometry & kinematics engine (new)

**Today (verified by reading code):** nothing reads existing index entries and writes derived values back (only `upload_documents`, no merge); `/corners` returns four video frames and stores nothing; `homography_routine` is standalone; `tracking.py`/`motion.py` report pixel-unit speed only (`mean_speed_px_s`, flow magnitude) and persist nothing derived. No distance/offset-from-corner or ground-unit kinematics exists anywhere.

**Goal:** after a published spatial revision exists, a batch/backfill engine derives per-frame and per-object quantities relative to the survey frame and writes them back as new revisions + idempotent index merges.

**Module:** `core/geospatial/postprocess/`
- `survey_frame.py` — builds and persists a **SurveyFrame** per video/revision: the four corners (from `survey_corners` telemetry or visual mode, or from registered keyframes), their order/labels, the local metric origin and axes (`east/north` for telemetry; heading-relative for visual mode), the ground scale (m/px) with its **source and validity**, and the homography image→survey plane (reusing `geometry.estimate_homography`/`map_points`).
- `offsets.py` — for each frame (camera/scene point) and each tracked object: signed offsets along survey axes, distance to each of the four corners and to the rectangle edges/centre, normalised position (0–1) inside the rectangle, and which corner/quadrant is nearest. Metres only when scale is validated; otherwise normalised/relative units only.
- `kinematics.py` — from trajectories (`tracking.py`) plus `FrameTiming.video_offset_us`: speed, heading, acceleration, path length, dwell time, displacement from start; computed in survey-plane coordinates, **time from `video_offset_us` only**, with smoothing and gap handling (no differentiation across segment resets/cuts, no extrapolation). Camera kinematics (ground speed, heading) only from telemetry or aligned reconstruction, never from pixel flow alone.
- `publish.py` — writes results to DB (`FrameDerivedMetrics`, `ObjectDerivedMetrics`, immutable per revision) and enqueues index merges through the existing outbox (§10).

**New index fields (additive, with units and status in the name/metadata):** `survey_frame_id`, `corner_distance_m` (4-element or `corner_dist_bl/tl/tr/br_m`), `nearest_corner`, `survey_u`, `survey_v` (normalised), `offset_east_m`, `offset_north_m`, `object_speed_mps`, `object_heading_deg`, `kinematics_status`, `derived_metrics_revision`. Same rule as other fields: unit-bearing fields are null unless the scale is validated; relative-only results use `*_rel` fields.

**Index write path (prerequisite):** add merge-style updates (`merge_or_upload_documents`) alongside `upload_documents`, keyed by the video-scoped document id from §11.1, batched, retried, and idempotent by payload hash. A backfill iterator pages through an account/video's documents by `blob_path` prefix (not by SAS URL) and never rewrites unchanged documents.

**Rules**
1. Metric values require a validated scale (known telemetry GPS, aligned reconstruction, or a user-supplied ground-sample distance); otherwise publish relative units with `status=relative_only`. Never relabel pixels as metres.
2. Visual-mode corners are heading-relative, not north-aligned: mark `survey_frame` orientation as `relative` and do not mix with telemetry-based frames.
3. Offsets reference the published spatial revision; a new revision supersedes derived metrics (stale revisions not published).
4. Runs in the same worker/lease model as §6; cancellable; account-scoped; dry-run emits deterministic structure with null metrics.
5. Object kinematics are per-track, so track IDs must be stable per revision; re-tracking creates a new `derived_metrics_revision`.

**Chat grounding:** extend the frame context (§11.3), e.g. "Position: 12 % from the west edge, 63 % from the south edge; nearest corner: bottom-left (≈48 m, estimated)" and "Object 3 speed: 4.2 m/s (estimated)", with the status preserved by the consolidator.

---

## 11. Index & agent grounding

### 11.1 Fix first (prerequisite PR, independent of geospatial)
These were already documented in `docs/agentic_retrieval.md` §9.3:
1. **Doc IDs not video-scoped** (`{account}-{frame:04d}`) → frames from different videos overwrite each other. New: `account-{account_id}-video-{video_id}-frame-{frame_id}`.
2. **`path` stores the full SAS URL**, but the retrieval filter does `startswith(path, '<acct>/images/<video>/')`. New: store `blob_path = "{account_id}/images/{video_id}/frame{N}.jpg"`; generate SAS only at read time. Keep expiring SAS URLs out of identity and filter fields.
3. Populate the already-declared `video_id`, `frame_number`, `timestamp`, `location` fields in `build_document`.

Migration of existing documents: re-index per account (idempotent upsert under the new IDs, then delete old IDs). Needs a runbook step — and it interacts with PR #29's `core.azure.deploy` (`--recreate-index`).

### 11.2 New index fields
`video_offset_us`, `flight_offset_us`, `capture_time_utc`, `camera_location` (GeographyPoint), `scene_location` (GeographyPoint), `camera_altitude_m`, `camera_altitude_reference`, `height_above_reference_plane_m`, `altitude_gain_from_baseline_m`, `horizontal_status`, `vertical_status`, `horizontal_uncertainty_m`, `vertical_uncertainty_m`, `geometry_revision`, `spatial_segment_id`, `evidence_manifest_id` (+ `position_method`). Altitude stays separate from the horizontal point. Updating schema of an existing index = additive fields only (Azure Search allows adding fields without rebuild).

### 11.3 Chat grounding
Extend the frame context passed to agents (where `curated_frame_urls` and frame context are assembled), e.g.:

```
Frame 7
Video offset: 14.2 s
Camera position: unavailable
Scene-center position: estimated
Height above reference plane: 24.1 m
Gain from initial baseline: +5.7 m
Geometry revision: geo-4
```
Update agent instructions (`solution-accelerator/agents/*.yaml`; synced by `core.azure.deploy`) so the consolidator **preserves method/uncertainty** and never rewrites "scene-centre estimate" as "drone location". Add a regression test using a canned transcript.

---

## 12. API & UI

Endpoints (account-scoped, DRF):
```
POST /api/v1/videos/{video_id}/spatial-enrichment/        # schedule; returns job id
GET  /api/v1/videos/{video_id}/spatial-enrichment/        # run status
POST /api/v1/videos/{video_id}/spatial-context/           # anchors, flight offset, UTC start, calibration
GET  /api/v1/videos/{video_id}/trajectory/
GET  /api/v1/videos/{video_id}/frames/{frame_id}/spatial/
```
Orthomosaic endpoints (account-scoped):
```
POST /api/v1/orthomosaics/                      # create project: video_ids[], options, referencing mode
GET  /api/v1/orthomosaics/{id}/                 # status, georeferencing_status, quality report
POST /api/v1/orthomosaics/{id}/runs/            # (re)run -> job id
GET  /api/v1/orthomosaics/{id}/download/{asset} # orthophoto (COG), dsm, cameras, report
DELETE /api/v1/orthomosaics/{id}/runs/{run_id}  # cancel
```
Per-account auto-trigger mode: `disabled | timestamps_only | telemetry_only | local_visual_analysis | external_visual_analysis`. External mode requires an explicit account policy (permitted providers, image/video transfer, retention, request limits, cost controls).

UI (dvsa-platform console, `/console/videos`): independent badges — *Time: video-derived*; *Horizontal location: estimated/unavailable*; *Vertical result: metric/relative-only/unavailable*; *Enrichment: running/partial/completed*. Frame details show baseline, altitude reference, spatial method, evidence revision. (Frontend lives in the separate dvsa-platform repo → its own PR.)

---

## 13. Operational safeguards

| Failure | Behaviour |
|---|---|
| External provider unavailable | Retry with backoff; keep timestamps + existing published revision |
| Unknown reconstruction scale | Relative units only |
| No geographic anchors | Camera geo location null |
| Poor plane fit | Reject plane-based height |
| Reconstruction reset | New segment; no cross-segment interpolation |
| Budget exhausted | `partial`; resume only under policy |
| Indexing failure | Retry outbox only |
| Worker crash | Resume from persisted stage artifacts |
| Video/anchors changed | Supersede stale run |
| Account/video mismatch | Reject artifact access + publication |

Logging: artifact IDs only — no SAS tokens or provider credentials.

---

## 14. Delivery phases (each = reviewable branch/PR; confirm before push)

| Phase | Scope | Exit criteria |
|---|---|---|
| **P0 — Index integrity** | Video-scoped doc IDs, `blob_path`, populate declared fields; re-index runbook | Cross-video docs no longer collide; path filter matches; tests |
| **P1 — Timestamp foundation** | `apps/geospatial` skeleton, `FrameTiming`, probe + timing manifest, API exposes `video_offset_us`, `timestamps_only` mode, run/job table + worker skeleton | Every extracted frame has a reliable offset + explicit `unavailable` spatial status |
| **P2 — Telemetry enrichment** | Telemetry provider (reuse `survey_corners` parsing), clock sync, bounded interpolation, camera-vs-aircraft offsets | `telemetry`/`telemetry_interpolated` frames; gap/segment tests |
| **P3 — Altitude adapter** | Async provider contract, LingBot-Map wrapper (container/GPU), canonical frame manifest, quality checks | Height/gain published as relative metrics with baseline definition |
| **P4 — Visual geo alignment** | (a) Productionize `aerial_location/` as `providers/registration.py` with synthetic parity tests; keyframe selection; scene-only results (`ambiguous`/`abstained` → unavailable). (b) Geo-aligned camera reconstruction using registered keyframes as anchors | (a) scene-only estimates with uncertainty + provenance; (b) camera location only with validated alignment |
| **P5 — Retrieval integration** | Index fields, outbox, chat context, agent instructions, UI badges (platform repo) | Agents quote method/uncertainty; stale revisions never published |
| **P6 — Post-processing geometry & kinematics** (after P1–P4; index merge path can start after P0) | (a) Index merge/update path + backfill iterator. (b) `SurveyFrame` persisted from `/corners` output + homography/scale. (c) Corner/edge offsets for frames and tracked objects. (d) Ground-unit kinematics from trajectories with `video_offset_us` timing. (e) Publish via outbox; chat context | Derived metrics are revisioned, idempotent, unit-correct (relative when unscaled), and queryable from the index and chat |
| **P7 — Photogrammetry / multi-tour orthomosaic** (after P1–P2; benefits from P4a) | (a) Spike: NodeODM container + `pyodm` on a small fixture (frames from 2 tours), measure runtime/memory, accuracy with and without GPS. (b) Frame selector + EXIF injection from telemetry. (c) `photogrammetry` provider, `OrthomosaicProject/Run/Artifact` models, storage, endpoints, cancellation. (d) Feed camera poses into Path B; derive `SurveyFrame` from mosaic extent for P6. (e) Index/chat fields | One COG orthomosaic from N tours with an explicit `georeferencing_status`, provenance per input, and camera poses published as estimates with uncertainty |

Dependency: P0 should wait for/rebase on **PR #29**.

### Open questions for the user
1. OK to add migrations for the new `apps/geospatial` app (CLAUDE.md prohibits `db/migrations/` unless asked)?
2. Worker tech — is there already a queue (Celery/RQ/Azure Queue/Functions) in use for `ingest_video`? Which do we standardise on?
3. Where will the LingBot-Map script run (local GPU, Azure container, AML)? Is an external provider acceptable at all (policy/privacy)?
4. What telemetry formats do real users upload (DJI SRT only? CSV/GPX/MAVLink)?
5. Source of geographic anchors — user-supplied keyframes via `spatial-context`, or Perplexity/Video Indexer hints (low trust)?
6. Should P0 ship as a standalone PR before geospatial work (recommended), and who re-indexes existing accounts?
7. Do we keep `ImageEntity.location`/`timestamp` populated for back-compat, and in what format (`virtual_location` = camera location only)?
8. Defaults for thresholds: max interpolation gap, max uncertainty, baseline frame count.
9. Is `aerial_location/` intended to be committed to this repo (it is untracked), and as a package under `core/geospatial` or kept as a separate tool?
10. What georeferenced reference imagery will be available (GeoTIFFs, which regions) or which ArcGIS service is licensed for computational matching and derived-output retention?
11. Is the VLM proposal step acceptable at all, and if so via Azure Foundry only?
12. Is there a real aerial test image with known ground truth (e.g. `Landmark07.jpg`, which the README expects) for a real-imagery validation test?
13. Should the four corners from `/corners` (frames, not coordinates) be persisted as a `SurveyFrame`, and is a user-supplied ground-sample distance / survey dimensions acceptable as a scale source when no telemetry exists?
14. Which derived metrics matter first: distance/offset from each corner, normalised position in the rectangle, per-object speed/heading, or all of them?
15. Is a backfill over existing index entries acceptable (needs the video-scoped IDs from P0 first), and should it run automatically after each revision or on demand?
16. Where should object trajectories (`tracking.py`) be stored so kinematics can be recomputed (currently nothing persists them)?
17. Is AGPL-3.0 ODM acceptable as a separately deployed NodeODM service, and where does it run (local GPU box, Azure Container Apps/AKS, VM)? Any data-egress constraints?
18. Do the tours have synchronized telemetry (SRT/CSV) for EXIF injection, or must referencing come from GCPs / `aerial_location` registration?
19. What coverage/overlap do real tours have (single pass vs grid, altitude, camera model) — enough for ODM to succeed from video frames?
20. Expected scale: frames per tour, number of tours per mosaic, target GSD and acceptable runtime/cost?
21. Preferred output formats/storage: COG GeoTIFF in blob, plus DSM and point cloud, or orthomosaic only?
22. Should the orthomosaic also be exposed to the Azure Search index/agents (e.g. frame position on the mosaic), or only via the API/UI?

---

## 15. Test plan

Essential cases (from the proposal), mapped to layers:

- **Unit (`core/geospatial`):** VFR timestamps; non-zero first PTS; seeked extraction returns a different frame than requested; video starting after takeoff; missing UTC start; altitude baseline median vs exact first-frame offset; unscaled reconstruction; scene available / camera unavailable; sloped reference plane; missing provider intervals; reconstruction segment reset; bounded interpolation (no extrapolation, no cross-segment).
- **Integration (Django):** duplicate scheduling + worker restart (lease/idempotency); stale-revision publication → superseded; cross-account isolation on every endpoint; outbox retry without re-inference; dry-run never emits coordinates.
- **Index:** cross-video document identity; `blob_path` filter; search publication retry; schema additive update.
- **Registration (Path C):** synthetic parity tests — warp a known tile with a known homography and assert the recovered centre, footprint and `geometrically_supported`; unrelated tile → `abstained`; two distant near-equal tiles → `ambiguous`; repetitive texture rejected; non-RGB/non-uint8/missing-CRS rejected; result mapped to `scene_location` only (never `camera_location`); SIFT vs LightGlue backends recorded in provenance; no token in logs; network disabled unless policy allows. Real-imagery test only when fixtures exist (Q12).
- **Photogrammetry (P7):** frame selector (overlap/blur thresholds, per-tour ordering via `FrameTiming`); EXIF writer round-trips lat/lon/alt/time and omits untrusted positions; provider contract against a fake NodeODM (submit/status/results/cancel, failure, timeout, restart); fixture end-to-end on a tiny synthetic or public dataset producing a valid COG with expected CRS/extent; unreferenced input yields `georeferencing_status=unreferenced` and no geographic fields; changed input set creates a new revision; cross-account isolation; cancellation actually stops the container task.
- **Post-processing (P6):** corner/edge distances on a synthetic rectangle with known answers; heading-relative vs north-aligned frames not mixed; unscaled input yields relative fields only and null metre fields; speed/heading from a synthetic constant-velocity track against `video_offset_us` (incl. VFR and a segment reset: no derivative across the reset, no extrapolation); superseded revision not published; merge-update idempotency (same payload hash → no write) and retry; backfill pages by `blob_path` prefix and skips unchanged documents; cross-account isolation.
- **Agent grounding:** consolidator keeps "scene-centre estimate" distinct from "drone location".
- Commands per CLAUDE.md: `npm test` is the documented default, but this repo's backend tests are pytest (to confirm at P1 start); run tests after any logic change.

---

## 16. Risks

- Scale/orientation ambiguity in visual reconstruction → mitigated by relative-only publication (Path D).
- GPU cost/latency and data-transfer policy → account-level mode gating + budgets.
- Index re-key (P0) touches live data → runbook + dry-run + per-account rollout.
- Open knowledge-agent 401 and base `dvsa-index` blocker (from `docs/agentic_retrieval.md` / PR #29 notes) may affect chat verification of P5.
- Photogrammetry: video frames are rolling-shutter, compressed and often low-overlap, so ODM may fail or produce distorted mosaics; AGPL licensing; heavy CPU/GPU/RAM and long runtimes; multi-tour merges need a consistent spatial reference and similar lighting; unreferenced results must not be presented as geographic.
- Post-processing: index back-fill touches live data (needs P0 IDs + merge path, dry-run first); visual-mode corners are heading-relative so offsets are only comparable within one survey frame; ground-unit kinematics are only as good as the scale source; trajectories are not persisted today, so P6 depends on a storage decision (Q16).
- `aerial_location/` is unvalidated (never executed end-to-end); heavy GPU dependency stack; recall limited by reference-imagery coverage; imagery licensing/egress/cost; planar-homography error from relief and imagery date.
- Time bases: telemetry/video clock drift; cut/trim detection reliability.

---

## 17. Resume checklist

1. Confirm the answers to §14 open questions.
2. PR #29 and the chat/platform fixes are merged; branch from current `main` (this plan lives on `feat/virtual-location-timestamp`, PR #31).
3. Re-read in code before starting: `apps/videos/frame_extract.py`, `FrameExtractAPIView._sync_image_entities`, `ingest_video`, `core/azure/document.py`, `core/azure/index_schema.py`, `core/azure/survey_corners.py`, existing events/adapters/orchestration modules (verify names used in §4).
4. Restate P0 (or P1) scope to the user, get confirmation, then implement on a fresh branch, commit in small steps, **ask before push/PR**.
5. Update memory with the chosen phase order and decisions.
