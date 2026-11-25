# Vehicle tracking video: implementation plan (`/api/v1/videos/vehicles/`)

Status: **plan only, no code yet**. Branch `feat/vehicle-tracking` (off `main` at `cc7df34`).
Audience: any implementing model (Opus / Sonnet) or developer. Follow the stages in order; each stage
lists files, exact behaviour, and acceptance checks. Do not skip the "Rules" section.

---

## 0. Goal in one paragraph

Add a direct-tool endpoint `PUT /api/v1/videos/vehicles/` that takes `account_id` and `video_id`
(a `VideoEntity` pk), downloads that account's uploaded drone video, **detects vehicles, tracks them
with persistent IDs while compensating for drone (camera) motion, classifies each vehicle as
MOVING / STATIONARY / UNKNOWN**, renders an annotated **H.264 MP4** (boxes, ID labels, class, trails,
a counts panel), uploads it to blob storage at
`{account_id}/images/{video_pk}/vehicles/vehicles_tracked.mp4`, and returns a **read-SAS URL** for it.
The visual result should look like the reference output of the cattle project
(`https://github.com/xMKx/cattle-monitoring/blob/main/docs/video/halter_tracked.mp4`): every source
frame annotated, colour by movement state, ID above each box, a short ground-registered trail, and a
translucent panel with live counts.

---

## 1. Rules (read first)

1. **Licence: do not copy code from `xMKx/cattle-monitoring`.** The repository has **no licence file**
   (GitHub reports `license: null`), so its code is all-rights-reserved. This plan describes the
   *algorithms* (standard, published techniques: ByteTrack association, Kalman filtering,
   LK-flow + RANSAC global motion compensation, windowed least-squares velocity). Implement them from
   this document in our own code. Do not paste functions from that repo.
2. **Ultralytics is AGPL-3.0** (library and the official YOLO weights). This plan uses it as the default
   detector behind an interface (`VehicleDetector`) so it can be swapped. The project owner must accept
   that licence for a hosted service, or pick an alternative (see Open question Q1). Keep the import lazy
   so the rest of dvsa-api never imports `ultralytics`/`torch`.
3. Repo rules (`CLAUDE.md`): never modify `db/migrations/` (this feature needs **no** migrations);
   surgical changes; restate the plan to the user and get confirmation before writing code; commit in
   reviewable stages; **ask before pushing or opening a PR**; run tests after changing logic.
4. Commit message trailer: `Co-Authored-By: <the implementing model> <noreply@anthropic.com>`.
   PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
5. Do not commit the untracked files in the main checkout (`mainindexedvideo.mp4`, `generate_orthomosaic.py`,
   `orthomosaic.jpg`, `aerial_location/`, `local-startup-scripts/`, etc.). Never print secrets from
   `local-startup-scripts/local.ps1`.
6. This branch is **independent of PR #32 (orthomosaic)**. Do not import `apps/videos/orthomosaic_jobs.py`
   (it is not on `main`). Section 6 gives a self-contained job module modelled on it.
7. Windows is the dev platform. Do not use the `resource` module (Unix only). Use `psutil` if memory
   stats are wanted, or omit them.

---

## 2. What the reference project does (summary of `cattle-monitoring` stage 2)

Measured facts from its README and source (Nov 2026, `ultralytics==8.4.216`, `torch==2.15`, `opencv 5.1`):

| Aspect | Reference (cattle) | Our adaptation (vehicles) |
|---|---|---|
| Decode | OpenCV sequential `cap.read()`, timestamps from `CAP_PROP_POS_MSEC` (never `index/fps`; drone/phone video is often VFR). No random seeking (unreliable with B-frames). | Same. |
| Frame selection | Detect on frames chosen **by timestamp** at `target_fps` (30); every source frame is still rendered with the latest annotations. | Same, default `target_fps=15` (configurable; vehicles move faster than cows, see §4.3). |
| Detector | YOLOE-26-x with auto-mined **visual prompts**, 4480 px input (cows were ~20 px, top-down). Closed-set COCO YOLO had recall ≤ 0.19 on that footage. | Closed-set COCO YOLO (`car`, `truck`, `bus`, `motorcycle`, `bicycle`), 1280 px input, optional tiling for small vehicles. Visual prompts are out of scope (Q2). |
| Camera motion | Shi-Tomasi corners on a downscaled grey frame, **detections masked out**, pyramidal LK flow to previous frame, RANSAC **similarity** (`estimateAffinePartial2D`), chained to the first processed frame → `H_cur_to_ref`. Step flagged `ok=False` if < 40 inliers. | Same. Default `similarity`; `homography` option for oblique footage. |
| Tracking | Own ByteTrack-style tracker **in the reference (ground) plane**: Kalman CV filter on (cx,cy,w,h) with per-second units; stage 1 IoU with high-score dets; stage 2 low-score dets only continue confirmed tracks; tentative tracks need `min_hits=3` before getting a public ID; lost tracks re-found by centre distance in body lengths; Hungarian assignment (`scipy.optimize.linear_sum_assignment`). | Same design, vehicle parameters (§4.4). Adds a per-track **class vote**. |
| Activity | Per track: least-squares velocity over a 1.5 s window in ground coords, normalised by body length (BL/s). MOVING if speed ≥ 0.20 and lower 2σ bound > 0.10; STATIONARY if speed ≤ 0.10 and upper 2σ bound < 0.20; otherwise keep previous; 0.5 s dwell before switching; UNKNOWN if too little history or a failed camera registration in the window. | Same algorithm, vehicle thresholds (§4.5). |
| Render | Box + ID label per object coloured by state (BGR MOVING `(0,70,255)` orange, STATIONARY `(255,210,0)` cyan, UNKNOWN `(200,200,200)` grey); trail = last 2 s of ground positions mapped back into the current image; dark translucent panel top-left with time, counts, unique IDs, camera-registration warning. | Same look, plus class name in label (`car 12`) and per-class counts in panel. |
| Encode | ffmpeg pipe: `rawvideo bgr24 → libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart`, CFR at the source's nominal fps; each source frame written as many times as needed to cover its display interval (keeps VFR timing). Falls back to OpenCV `mp4v`. | Same, but `mp4v` is **not playable in browsers**; see §4.7 for the fallback order. |
| Output JSON | Video info, models/params, performance, summary (measured vs model estimates), per-track records, per-frame records. | Smaller: summary + per-track records uploaded as JSON; per-frame list omitted by default (size). |
| Speed | 13 s 1280×516 clip: 279 s on M4 Max; 94 % of time is detection at 4480 px. | COCO YOLO at 1280 px on an RTX-class GPU should be ≈ 20–40 ms/frame; CPU ≈ 0.3–1 s/frame. Long videos need a background job (§6). |

Known weaknesses the reference reports, which apply to us too: re-entry after leaving the frame gets a
new ID (no appearance re-ID); dense groups fragment IDs; thresholds tuned on one video; no metric
units (no ground-sample distance), so speeds are in **body lengths per second**, never m/s or km/h.

---

## 3. API contract

Route: `PUT /api/v1/videos/vehicles/` and `GET /api/v1/videos/vehicles/` (file `apps/videos/urls.py`,
name `vehicles`). Permission: `IsAuthenticated` (JWT, same as `/corners/`). Parsers:
`MultiPartParser, FormParser, JSONParser`.

### PUT request body

| Field | Type | Required | Meaning |
|---|---|---|---|
| `account_id` | string/int | yes | Account that owns the video. |
| `video_id` | int | yes | `VideoEntity` pk; must belong to `account_id`. |
| `wait` | bool | no, default `false` | Run inline and return the finished result (200) in this response. |
| `force` | bool | no, default `false` | Regenerate even if the MP4 already exists. |
| `start_s`, `end_s` | float | no | Process only this time range (seconds). `end_s` is capped by `DVSA_VEHICLES_MAX_SECONDS`. |

Booleans accept `true/1/yes/on` (case-insensitive), as JSON or form fields.

### PUT responses

| Case | Status | Body |
|---|---|---|
| Missing `account_id` or `video_id` | 400 | `{"error": "account_id is required"}` / `{"error": "video_id is required"}` |
| Video not found for account | 404 | `{"error": "video not found"}` |
| Result already exists and not `force` | 200 | `{account_id, video_pk, state: "completed", downloadUrl, reportUrl}` |
| Job started / already queued or running | 202 | `{account_id, video_pk, state: "queued"|"running"}` |
| `wait=true` and success | 200 | `{account_id, video_pk, state: "completed", downloadUrl, reportUrl, summary}` |
| `wait=true` and detector runtime missing (no `ultralytics`/`torch`, weights cannot load) | 503 | `{..., state: "failed", unavailable: true, reason}` |
| `wait=true` and other failure | 500 | `{..., state: "failed", unavailable: false, reason}` |

`downloadUrl` = read-SAS URL of the MP4 (`content_type="video/mp4"`). `reportUrl` = read-SAS URL of
`vehicles_tracking.json`. SAS lifetime: `DVSA_VEHICLES_SAS_HOURS` (default 24).

### GET `?account_id=&video_id=`

Always 200 (400/404 as above for bad input):
`{account_id, video_pk, state: "none"|"queued"|"running"|"completed"|"failed", downloadUrl?, reportUrl?, reason?, unavailable?, progress?}`.
`completed` is resolved from **blob existence** (survives restarts). `progress` is
`{"processed_s": float, "duration_s": float}` while running.

### Summary object (`summary`, also the top of the JSON report)

```json
{
  "video": {"width": 1920, "height": 1080, "duration_s": 42.1, "fps_nominal": 29.97, "variable_frame_rate": false,
            "frames_decoded": 1262, "frames_processed": 631},
  "vehicles": {
    "unique_track_ids": 37, "unique_track_ids_lasting_ge_1s": 30,
    "peak_simultaneous_tracked": 14, "peak_at_t": 12.4,
    "by_class": {"car": 29, "truck": 5, "bus": 1, "motorcycle": 2, "bicycle": 0},
    "tracks_by_dominant_state": {"moving": 21, "stationary": 14, "unknown": 2},
    "camera_registration_failures": 0,
    "note": "Counts are model estimates, not validated. IDs over-count when tracks fragment; speeds are in body lengths per second (no metric scale)."
  },
  "models": {"detector": {"weights": "yolo26m.pt", "imgsz": 1280, "classes": ["car", "..."], "device": "cuda:0"},
             "tracker": {"...": "TrackerConfig as dict"}, "motion_model": "similarity", "activity": {"...": "ActivityConfig as dict"}},
  "timing_s": {"download": 3.1, "detection": 25.0, "motion": 4.2, "tracking": 0.6, "activity": 0.5,
               "render_encode": 6.0, "upload": 2.2, "total": 43.0}
}
```

---

## 4. Algorithm specification (implement exactly; parameters live in dataclasses)

All coordinates are pixels. "Image" = current frame pixels. "Ref" (ground) = pixels of the **first
processed frame's** image plane. Boxes are `xyxy` `float64` arrays of shape `(N, 4)`.

### 4.1 Video I/O (`core/vehicles/video_io.py`)

- `probe_video(path) -> VideoInfo` dataclass: `path, width, height, frame_count, duration_s, fps_nominal,
  fps_average, codec, variable_frame_rate`. Use `ffprobe` when found (see §4.7 for locating binaries):
  `ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=width,height,codec_name,r_frame_rate,avg_frame_rate,nb_read_packets,duration -of json PATH`
  (60 s timeout). `fps_nominal = Fraction(r_frame_rate)`; `frame_count = nb_read_packets`;
  `fps_average = frame_count / duration`; `variable_frame_rate = |avg - nominal| > 1 % of nominal`.
  Fallback: OpenCV `CAP_PROP_FRAME_COUNT`/`CAP_PROP_FPS`. Raise `ValueError` if the file cannot be opened.
- `iter_frames(path) -> Iterator[(index, t_seconds, frame_bgr)]`: sequential `cap.read()`; `t =
  cap.get(cv2.CAP_PROP_POS_MSEC) / 1000` **after** `read()`. Release the capture in `finally`.
- `FrameSelector(target_fps, source_fps)`: callable `(t) -> bool`. `interval = 1/target_fps` (0 if
  `target_fps <= 0` → select all). Tolerance `0.5/source_fps`. Select when first call, interval 0, or
  `t - last >= interval - tol`; then `last = t`.
- `Mp4Writer(path, width, height, fps)`: CFR H.264 writer.
  - ffmpeg command: `[ffmpeg, -v, error, -y, -f, rawvideo, -pix_fmt, bgr24, -s, WxH, -r, f"{fps:.6f}", -i, -, -c:v, libx264, -preset, veryfast, -crf, 23, -pix_fmt, yuv420p, -movflags, +faststart, PATH]`
    via `subprocess.Popen(stdin=PIPE)`. Width and height must be **even** for yuv420p: round the render
    size down to even numbers before creating the writer.
  - `write(frame, t_end)`: emit the frame until `n_written >= max(n_written + 1, round(t_end * fps))`.
    `t_end` is the timestamp (relative to the first rendered frame) of the **next** frame. This keeps VFR
    timing and total duration.
  - `close()`: close stdin, `wait()` with timeout 300 s, raise `RuntimeError` on non-zero return code
    (include ffmpeg stderr tail).
  - Must work as a context manager or be closed in `finally` so a failed run never leaves ffmpeg running.

### 4.2 Camera motion (`core/vehicles/motion.py`)

`CameraMotionEstimator(model="similarity", max_features=1500, ransac_px=1.0, min_inliers=40, max_width=1280, border_px=8, mask_pad=0.5)`.
`update(frame_bgr, exclude_boxes) -> MotionStep(H_prev_to_cur, H_cur_to_ref, ok, n_features, n_inliers, residual_px)`:

1. Grey + downscale to `scale = min(1, max_width / width)`.
2. Feature mask: 255 everywhere except an 8 px border, and **zero inside every detection box padded by
   50 % of its size** (vehicles move independently of the ground; they must not drive the estimate).
   Note: the mask is built from the *current* frame's boxes but used when this frame becomes the
   *previous* frame of the next update.
3. First call: store and return identity, `ok=True`.
4. `cv2.goodFeaturesToTrack(prev, maxCorners=1500, qualityLevel=0.005, minDistance=8, mask=prev_mask, blockSize=7)`.
   Need ≥ `min_inliers` points.
5. `cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21,21), maxLevel=3, criteria=(EPS|COUNT, 30, 0.01))`; keep `status==1`.
6. Fit: `similarity` → `cv2.estimateAffinePartial2D`; `affine` → `cv2.estimateAffine2D`; `homography` →
   `cv2.findHomography(..., cv2.RANSAC, ransac_px, maxIters=2000, confidence=0.995)`; affine variants
   use `method=cv2.RANSAC, ransacReprojThreshold=ransac_px, maxIters=2000, confidence=0.995` and are
   lifted to 3×3. `none` → identity, ok.
7. `ok = fit succeeded and inliers >= min_inliers`. If ok, rescale to full resolution:
   `A = inv(S) @ A @ S`, `S = diag(scale, scale, 1)`. If not ok, `A = I` (assume no motion, flag it).
8. `H_cur_to_ref = H_cur_to_ref @ inv(A)`, normalise by `[2,2]`.

Helpers: `apply_h(H, pts(N,2)) -> (N,2)`; `transform_boxes(H, boxes)` = axis-aligned bbox of the four
transformed corners.

Road scenes caveat: a road is planar, which suits this model. Large vehicles filling most of the
frame starve the estimator; that is expected to produce `ok=False` steps and UNKNOWN states, not crashes.

### 4.3 Detector (`core/vehicles/detector.py`)

Interface (a `Protocol`), so tests can inject a fake:

```python
class VehicleDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (boxes_xyxy float64 (N,4), scores float64 (N,), class_names list[str] length N)."""
    def describe(self) -> dict: ...
```

Default implementation `UltralyticsVehicleDetector(DetectorConfig)`:

- `DetectorConfig`: `weights="yolo26m.pt"` (verify the name exists for the installed ultralytics; fall
  back to `yolo11m.pt`), `device=None` (auto: `cuda:0` → `mps` → `cpu`), `imgsz=1280`, `conf=0.10`
  (low on purpose: the tracker's second stage uses 0.10–0.25 detections), `classes=("car","truck","bus","motorcycle","bicycle")`,
  `max_det=1000`, `half=True` on CUDA only, `tile=False`, `tile_size=1280`, `tile_overlap=0.2`, `merge_iou=0.5`.
- Import `ultralytics` lazily inside `__init__`. Wrap `ImportError` and weight-load errors in
  `VehiclesUnavailable(RuntimeError)` (this drives the 503).
- Map wanted class names to model ids via `model.names` (case-insensitive). Raise `VehiclesUnavailable`
  if none match.
- `model.predict([frame], imgsz, conf, classes=ids, agnostic_nms=True, max_det, device, half, verbose=False)`;
  read `result.boxes.xyxy`, `.conf`, `.cls` to numpy.
- Warm up once on a zero image of the video's frame shape.
- Optional tiling (`tile=True`): run overlapping `tile_size` crops in addition to the full frame, offset
  boxes back, drop tile boxes touching an interior tile edge (within 2 px), merge everything with greedy
  NMS at `merge_iou`. Write this as small pure-numpy functions (`tile_windows`, `touches_interior_edge`,
  `box_iou`, `nms`) so they are unit-testable.
- Store weights in `DVSA_VEHICLES_MODEL_DIR` (default `<BASE_DIR>/models/`), git-ignored. Ultralytics
  downloads official weights on first use; document that the first run needs internet.

Why COCO first: vehicles are COCO classes and are much larger than the reference project's cows in
typical low-altitude footage. For high-altitude nadir footage COCO recall drops sharply (domain shift);
see Q2 for VisDrone-trained weights or visual prompts.

### 4.4 Tracker (`core/vehicles/tracker.py`, pure numpy + scipy)

Tracks live in **ref coordinates** (`boxes_ref = transform_boxes(H_cur_to_ref, boxes_img)`), so drone
motion is not object motion.

`TrackerConfig` (vehicle defaults; all times in seconds):

| Field | Default | Meaning |
|---|---|---|
| `high_thresh` | 0.25 | Stage-1 detections. |
| `low_thresh` | 0.10 | Stage-2 detections may only continue existing confirmed tracks. |
| `new_track_thresh` | 0.30 | Minimum score to start a tentative track. |
| `match_iou` | 0.20 | Stage-1 IoU gate (active tracks). |
| `low_match_iou` | 0.30 | Stage-2 IoU gate. |
| `tentative_iou` | 0.30 | Tentative-track IoU gate. |
| `refind_dist_bl` | 1.0 | Lost-track re-find gate in body lengths … |
| `refind_growth_bl_s` | 1.5 | … growing this much per second lost … |
| `refind_max_bl` | 4.0 | … up to this cap. |
| `min_hits` | 3 | Consecutive matches before a public ID is assigned. |
| `max_lost_s` | 2.0 | Keep a lost track this long. |
| `lost_velocity_decay_s` | 5.0 | Predicted velocity of a lost track decays with this time constant (vehicles keep moving). |

Kalman filter (`KalmanBox`): state `[cx, cy, w, h, vcx, vcy, vw, vh]`, velocities per **second**.
`predict(dt, velocity_scale)`: `F = I; F[:4,4:] = I*dt`; multiply velocities by `velocity_scale`
first; process noise `Q = diag((std_pos*s)^2 x4, (std_vel*s)^2 x4) * (dt*30)` with `s = max(w,h,1)`,
`std_pos = 1/20`, `std_vel = 30/160`. Initial covariance `diag((2*std_pos*s)^2 x4, (10*std_vel*s)^2 x4)`.
`update(box)`: measurement `[cx,cy,w,h]`, `R = diag((std_pos*s)^2 x4)`, standard KF update. Skip
predict when `dt <= 0`.

Track record: `uid` (internal), `id` (public, assigned on confirmation, sequential from 1), `state`
(`tentative|confirmed|lost|removed`), `hits`, `first_t`, `last_t`, `lost_since`, `gaps[(t0,t1)]`,
`end_reason`, `obs[Observation(t, frame, box_ref, box_img, conf, stage, class_name)]`, and
`class_votes: Counter`. `class_name` property = most common vote (ties: highest summed score).

`update(t, frame, boxes_ref, boxes_img, scores, classes, in_view) -> list[(Track, det_index)]` per step:

1. `dt = t - previous t` (0 on first call). Predict all tracks; lost tracks use
   `velocity_scale = exp(-dt / lost_velocity_decay_s)`.
2. Split detections: `high = score >= high_thresh`, `low = low_thresh <= score < high_thresh`.
3. Stage 1: IoU-match confirmed+lost tracks to `high` (cost `1 - IoU`, gate `1 - match_iou`, Hungarian).
4. Stage 2: remaining **confirmed** tracks vs `low` at `low_match_iou`.
5. Stage 3: tentative tracks vs remaining `high` at `tentative_iou`.
6. Stage 4 (re-find): remaining confirmed/lost tracks vs remaining `high` by centre distance normalised
   by each track's size (`max(w,h)`) and its own gate `min(refind_max_bl, refind_dist_bl + refind_growth_bl_s * (t - last_t))`;
   Hungarian with threshold 1.0.
7. Apply matches: KF update, `hits += 1`, append observation and class vote; a lost track that matches
   gets `gaps.append((lost_since, t))` and returns to confirmed; a tentative track reaching `min_hits` is
   confirmed and gets the next public id. Return confirmed matches only.
8. Unmatched tentative tracks → removed (`end_reason="unconfirmed"`). Unmatched confirmed → lost
   (`lost_since = last_t`). Any track with `t - last_t > max_lost_s` → removed with
   `end_reason = "left_view"` if `in_view(track)` is false, else `"lost"`.
   `in_view` maps the KF centre back into the image with `inv(H_cur_to_ref)` and checks it is within
   the frame ±5 px.
9. Unmatched `high` detections with `score >= new_track_thresh` start tentative tracks.
10. Move removed tracks that ever had a public id to `finished`.

`finalize()`: end remaining tracks (`end_of_video` or `lost_at_end`), return confirmed tracks sorted by id.

Gating note for vehicles: IoU association needs the box to move less than about half its length between
processed frames. A car at 20 m/s seen at ~5 px/m moves ~100 px/s, so a 90 px car moves ~0.07 BL per
frame at 15 fps (fine). Fast/high-zoom footage needs `target_fps` 30 or 0 (every frame). The
Kalman prediction also absorbs constant velocity.

### 4.5 Activity (`core/vehicles/activity.py`)

`ActivityConfig`: `window_s=1.0`, `min_span_s=0.5`, `min_points=5`, `moving_bl_s=0.25`,
`stationary_bl_s=0.08`, `z=2.0`, `min_dwell_s=0.5`, `min_se_bl_s=0.01`.
(A 4.5 m car crawling at 1.1 m/s ≈ 0.25 BL/s; a parked car's measured noise should sit well below 0.08.)

Per confirmed matched track each processed frame:

1. Take its observations with `t > now - window_s`; centre points in ref coords; body length = median of
   `max(w, h)` of those ref boxes.
2. If any camera step in the window had `ok=False` → UNKNOWN (`reason="camera_registration_failed"`).
3. If fewer than `min_points` or span `< min_span_s` → UNKNOWN (`insufficient_history`).
4. Least squares per axis: `tc = t - mean(t)`, `v = Σ tc·(xy - mean) / Σ tc²`; residual sigma with
   `dof = max(1, 2n - 4)`; `se = sigma / sqrt(Σ tc²)`; `speed = |v|`; divide speed and se by body length.
   `heading_deg = degrees(atan2(vy, vx))` (image convention, 0 = right, 90 = down).
5. `se = max(se, min_se_bl_s)`. Candidate MOVING if `speed >= moving` and `speed - z·se > stationary`;
   STATIONARY if `speed <= stationary` and `speed + z·se < moving`; else no candidate (keep state).
6. Hysteresis: from UNKNOWN accept the candidate at once; a change between MOVING and STATIONARY must
   persist `min_dwell_s` before it is accepted.
7. `confidence = Φ(margin / se)` where margin is the distance from the threshold midpoint. Label it
   "not a calibrated probability" in the JSON.

### 4.6 Rendering (`core/vehicles/render.py`)

`draw_frame(frame, items: list[VehicleDraw], panel: PanelInfo, scale=1.0) -> np.ndarray` (returns a copy):

- `VehicleDraw(id, class_name, box_img, state, trail_img | None)`.
- Unit `u = max(w, h) / 1280`. Line thickness `max(1, round(2*u))`. Font `cv2.FONT_HERSHEY_SIMPLEX`,
  label scale `0.45*u`, `cv2.LINE_AA`.
- Colours (BGR): MOVING `(0, 70, 255)`, STATIONARY `(255, 210, 0)`, UNKNOWN `(200, 200, 200)`.
- Draw all trails first (polyline, state colour), then boxes, then labels `f"{class_name} {id}"` on a
  filled black rectangle above the box (below it if there is no room).
- Trail: the track's ref-coordinate centres from the last `trail_s=2.0` s mapped to the current image
  with `inv(H_cur_to_ref)`; only drawn with ≥ 2 points.
- Panel top-left at `(6u, 6u)`: darken the region to 35 % then write lines:
  1. `t MM:SS.ss / MM:SS.ss   frame N`
  2. `vehicles tracked in view: N   (raw detections: M)`
  3. `MOVING n` (orange), 4. `STATIONARY n` (cyan), 5. `UNKNOWN n` (grey)
  6. `car a  truck b  bus c  moto d  bike e` (in-view counts by class)
  7. `unique IDs so far: K (estimate, unvalidated)`
  8. red `camera registration FAILED this frame` only when the step failed.
- Output size: `render_scale` (default 1.0), then cap the long edge at `DVSA_VEHICLES_MAX_RENDER_EDGE`
  (default 1920) to bound file size; force even width/height.

### 4.7 ffmpeg/ffprobe discovery (`core/vehicles/ffmpeg.py`)

Order: `settings.DVSA_FFMPEG` (explicit path) → `shutil.which("ffmpeg")` → `imageio_ffmpeg.get_ffmpeg_exe()`
(pip `imageio-ffmpeg`, BSD-2, bundles a static ffmpeg; add to requirements). ffprobe: `DVSA_FFPROBE` →
`which("ffprobe")` → none (fall back to OpenCV probing; `imageio-ffmpeg` has no ffprobe).
If no ffmpeg at all: write with OpenCV `cv2.VideoWriter(..., fourcc "avc1")`; if that fails to open,
use `"mp4v"` and set `summary.output.browser_playable = false` with a log warning (mp4v MPEG-4 Part 2
does not play in Chrome/Edge/Safari `<video>`). On this dev box ffmpeg 9.0.2 is on `PATH`
(`C:\Users\ravib\ffmpeg\ffmpeg-9.0.2-essentials_build\bin`).

### 4.8 Pipeline orchestration (`core/vehicles/pipeline.py`)

```python
@dataclass
class VehiclePipelineConfig:
    target_fps: float = 15.0
    start_s: float = 0.0
    end_s: float | None = None
    max_seconds: float = 300.0
    motion_model: str = "similarity"
    render_scale: float = 1.0
    max_render_edge: int = 1920
    trail_s: float = 2.0
    trajectory_hz: float = 5.0
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)

def run_vehicle_pipeline(video_path: str, out_dir: str, cfg: VehiclePipelineConfig,
                         make_detector: Callable[[np.ndarray], VehicleDetector],
                         progress: Callable[[float, float], None] | None = None) -> dict:
    """Returns the report dict; writes out_dir/vehicles_tracked.mp4 and out_dir/vehicles_tracking.json."""
```

Single sequential pass over `iter_frames`:

1. Skip frames before `start_s`; stop after `min(end_s, start_s + max_seconds)`.
2. On the first kept frame create the detector via `make_detector(frame)` (lazy; allows warm-up on the
   real shape) and the writer (render size from §4.6).
3. If `selector(t)`: detect → camera motion update with boxes scoring ≥ `low_thresh` as the mask →
   `boxes_ref` → tracker update → activity per matched track → build the `VehicleDraw` list and
   `PanelInfo` → append a compact per-frame counts record (kept in memory for the timeline only).
4. Render **every** kept source frame with the latest draw state (the panel time/frame number updates
   each frame). Use a one-frame delay: hold the rendered frame as `pending`, write it when the next
   frame's timestamp is known (`writer.write(pending, t_next - t_first)`); after the loop write the last
   one with `t_last - t_first + 1/fps_nominal`.
5. Every ~2 s of video call `progress(t - t_first, planned_duration)`.
6. Time each phase with `time.perf_counter()` (`decode, detection, motion, tracking, activity, render_encode`).
7. Close the writer in `finally`.
8. Build the report (§3 summary + `tracks` list). Per-track record: `id, class, class_votes, first_t,
   last_t, duration_s, observed_frames, observed_fraction, durations_s {moving, stationary, unknown},
   dominant_state, state_switches, body_length_ref_px, path_length_bl, net_displacement_bl,
   median_speed_bl_s, max_speed_bl_s, trajectory_ref_px [[t, x, y], …] at trajectory_hz, uncertainty
   {interruptions, interrupted_s, low_confidence_matches, refinds, mean_conf, end_reason}`. Path length uses
   0.5 s median-binned positions (raw frame-to-frame sums measure jitter).

No Django imports in `core/vehicles/` (keeps it unit-testable and reusable from a CLI).

### 4.9 Optional CLI for local runs (`core/vehicles/__main__.py`)

`python -m core.vehicles VIDEO --out-dir DIR [--start S] [--end S] [--target-fps F] [--weights W] [--imgsz N] [--device D] [--tile]`
Prints the summary. Used for the live smoke test (§8) without Azure.

---

## 5. Settings (`config/settings/base.py`, appended block; all `os.environ.get` strings)

```python
# Vehicle tracking video (`/videos/vehicles/`). Detector needs `ultralytics` + `torch`
# (optional; see requirements/vehicles.txt). Without them the endpoint returns 503.
DVSA_VEHICLES_WEIGHTS = os.environ.get("DVSA_VEHICLES_WEIGHTS", "yolo26m.pt")
DVSA_VEHICLES_MODEL_DIR = os.environ.get("DVSA_VEHICLES_MODEL_DIR", str(BASE_DIR / "models"))
DVSA_VEHICLES_DEVICE = os.environ.get("DVSA_VEHICLES_DEVICE", "")          # "" = auto
DVSA_VEHICLES_IMGSZ = os.environ.get("DVSA_VEHICLES_IMGSZ", "1280")
DVSA_VEHICLES_CONF = os.environ.get("DVSA_VEHICLES_CONF", "0.10")
DVSA_VEHICLES_TILE = os.environ.get("DVSA_VEHICLES_TILE", "false")
DVSA_VEHICLES_TARGET_FPS = os.environ.get("DVSA_VEHICLES_TARGET_FPS", "15")
DVSA_VEHICLES_MAX_SECONDS = os.environ.get("DVSA_VEHICLES_MAX_SECONDS", "300")
DVSA_VEHICLES_MOTION_MODEL = os.environ.get("DVSA_VEHICLES_MOTION_MODEL", "similarity")
DVSA_VEHICLES_MAX_RENDER_EDGE = os.environ.get("DVSA_VEHICLES_MAX_RENDER_EDGE", "1920")
DVSA_VEHICLES_MAX_JOBS = os.environ.get("DVSA_VEHICLES_MAX_JOBS", "1")     # concurrent GPU jobs
DVSA_VEHICLES_SAS_HOURS = os.environ.get("DVSA_VEHICLES_SAS_HOURS", "24")
DVSA_FFMPEG = os.environ.get("DVSA_FFMPEG", "")
DVSA_FFPROBE = os.environ.get("DVSA_FFPROBE", "")
```

`BASE_DIR` (a `Path`) and `os` are already defined at the top of `base.py`. `config/settings/testing.py`
needs no change (no automatic job on upload).

Requirements:
- `requirements/base.txt`: add `imageio-ffmpeg>=0.5` (small, pure-wheel ffmpeg fallback).
- New `requirements/vehicles.txt` (optional install, documented in the README section):
  ```
  -r base.txt
  ultralytics>=8.4,<8.5    # AGPL-3.0, see docs/vehicle_tracking_plan.md Q1
  # torch: install the build matching the GPU. RTX 50-series (Blackwell, sm_120) needs CUDA 12.8+ wheels:
  #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
  Do **not** add ultralytics/torch to `base.txt` (CI and other features must not pull a 2+ GB stack).
- Add `models/` to `.gitignore` (it is not ignored today; weights must never be committed).

---

## 6. Background job + view

### 6.1 `apps/videos/vehicle_jobs.py` (self-contained; mirrors the orthomosaic job pattern from PR #32)

```python
QUEUED, RUNNING, COMPLETED, FAILED = "queued", "running", "completed", "failed"
def video_blob_name(account_id, video_pk) -> str:  # f"{account_id}/images/{video_pk}/vehicles/vehicles_tracked.mp4"
def report_blob_name(account_id, video_pk) -> str: # f"{account_id}/images/{video_pk}/vehicles/vehicles_tracking.json"
def get_state(account_id, video_pk) -> dict        # Django cache key f"vehicles:{account_id}:{video_pk}", default {"state": "none"}
def status(account_id, video_pk, config=None) -> dict   # cache state; if MP4 blob exists and not queued/running -> completed + fresh SAS urls
def build_config() -> VehiclePipelineConfig        # from DVSA_VEHICLES_* settings; start_s/end_s overrides passed in
def make_detector_factory() -> Callable            # returns lambda frame: UltralyticsVehicleDetector(DetectorConfig(...), warmup_shape=frame.shape)
def run_job(account_id, video_pk, source_url, start_s=None, end_s=None) -> dict   # NEVER raises
def start_job(account_id, video_pk, source_url, *, force=False, wait=False, start_s=None, end_s=None, config=None) -> dict
```

`run_job` steps (inside a module-level `BoundedSemaphore(DVSA_VEHICLES_MAX_JOBS)`):

1. state RUNNING.
2. **Check the detector runtime before downloading**: `importlib.util.find_spec("ultralytics")` and
   `find_spec("torch")`; if missing raise `VehiclesUnavailable("ultralytics/torch not installed")`.
3. `video_path = azure_blob.download_blob_to_temp(source_url)`; `work_dir = tempfile.mkdtemp(prefix="dvsa-vehicles-")`.
4. `report = run_vehicle_pipeline(video_path, work_dir, cfg, make_detector, progress=...)`; progress
   callback writes `{"state": RUNNING, "progress": {...}}` to the cache at most every 2 s.
5. Upload: MP4 with `azure_blob.put_blob_and_sas(config, video_blob_name(...), mp4_bytes,
   content_type="video/mp4", ttl_hours=SAS_HOURS)`; JSON (report without per-frame data) with
   `content_type="application/json"`. Read the MP4 as bytes (files are tens of MB; acceptable) or pass an
   open file handle (`upload_blob` accepts streams; preferred for long videos).
6. state COMPLETED with `downloadUrl`, `reportUrl`, `summary`.
7. `except Exception`: state FAILED, `reason=str(exc)[:500]`,
   `unavailable = isinstance(exc, VehiclesUnavailable)`. Log with `logger.warning`.
8. `finally`: delete temp video and work dir (`shutil.rmtree(ignore_errors=True)`), remove the key from the
   in-process `_active` set.

`start_job`: if not `force` and `status()` is completed/queued/running → return it. Guard duplicate
runs per `(account_id, video_pk)` with a `threading.Lock` + `_active` set. Set QUEUED. `wait=True` →
call `run_job` inline and return its state. Else start `threading.Thread(daemon=True,
name=f"vehicles-{video_pk}")`.

Known limitation (document in the docstring): state lives in the Django cache (LocMem per process by
default), and daemon threads die with the process; the blob remains the source of truth for "completed".
A Celery task is the production path (celery is already in `requirements/base.txt`), out of scope here (Q4).

### 6.2 View (`apps/videos/views.py`, appended) and URL

```python
class VehicleTrackingAPIView(APIView):
    """Track vehicles in an account's video and return an annotated MP4 (read-SAS URL). Docstring
    must describe the contract in section 3 and the blob layout."""
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    permission_classes = [permissions.IsAuthenticated]
    def put(self, request, pk=None, format=None): ...
    def get(self, request, pk=None, format=None): ...
```

- Resolve the entity with `VideoEntity.objects.filter(pk=video_id, account_id=account_id).first()`;
  404 if missing; `source_url = entity.sas_url`.
- Parse `start_s`/`end_s` with `float()`; 400 `{"error": "start_s/end_s must be numbers"}` on failure;
  400 if `end_s <= start_s`.
- Map the job state to status codes exactly as in section 3.
- Wrap only unexpected exceptions in 500 with `logger.exception("vehicle tracking request failed")`.
- `apps/videos/urls.py`: after the `extract-frames/` line add
  `path('vehicles/', views.VehicleTrackingAPIView.as_view(), name='vehicles'),` with a one-line comment
  like its neighbours.
- If this branch is merged after PR #32, both PRs append to `views.py`, `urls.py`, `base.py`: resolve
  the merge conflict by keeping both blocks.

---

## 7. Tests (`tests/test_vehicles.py`, fully offline, no torch/ultralytics/Azure)

Run: `~/dvsa-venv/Scripts/python -m pytest tests/test_vehicles.py --no-cov -q`, then the full suite
`~/dvsa-venv/Scripts/python -m pytest -q --no-cov --ignore=tests/test_photogrammetry_fixtures.py`
(that file is an untracked leftover in the main checkout; it is not on this branch). Baseline on `main`:
all tests pass; the full suite must still pass.

Synthetic fixture (`tests/vehicle_fixtures.py`):
- `make_scene_video(path, seconds=4, fps=30, size=(640, 360), pan_px_s=(40, 10), vehicles=[...])`:
  a large random-texture "ground" image (Gaussian-blurred noise, histogram-equalised, so LK has features);
  a camera window panning across it; vehicles = filled rectangles with a darker roof stripe, each with a
  ground-space start position and velocity (px/s), some with velocity 0 (parked). Write with
  `cv2.VideoWriter(mp4v)` (decoding mp4v with OpenCV is fine). Return ground-truth image-space boxes per
  frame.
- `FakeDetector(gt_boxes_by_frame, class_name="car", score=0.9, drop_every=None, jitter_px=0.0)`:
  implements `VehicleDetector` from ground truth; `drop_every=k` drops a detection every k frames to
  exercise lost/refind.

Unit tests (minimum set; each a separate test function):
1. `FrameSelector` at 30 → 15 fps picks every second frame; a frame after a gap is always picked.
2. `Mp4Writer.write` timing: with `_emit` monkeypatched, VFR timestamps produce `round(duration*fps)` frames.
3. `CameraMotionEstimator` recovers a known rotation 1.5°, scale 1.03, shift (6, −4) on a texture within
   0.3 px for `similarity` and `homography`; `H_cur_to_ref` maps back to the first frame.
4. Motion with a big moving rectangle masked out still recovers the background shift (mask works).
5. `transform_boxes` identity and translation.
6. Tracker: two vehicles crossing with constant velocity keep their IDs (IDs 1 and 2, no swap).
7. Tracker: a one-frame false positive never gets a public ID (`min_hits`).
8. Tracker: a vehicle missing for 0.5 s (detections dropped) is re-found with the same ID; one missing
   longer than `max_lost_s` gets a new ID.
9. Tracker: low-score (0.15) detections continue a confirmed track but never start one.
10. Tracker class vote: 4 × "car" + 1 × "truck" → `class_name == "car"`.
11. Activity: a stationary point with 1 px jitter → STATIONARY; a point moving 1 BL/s → MOVING with
    heading ≈ the true direction; under 5 points → UNKNOWN; camera failure in the window → UNKNOWN;
    a MOVING→STATIONARY change is delayed by `min_dwell_s`.
12. Tiling helpers: `tile_windows` covers the image; interior-edge boxes are dropped; `nms` keeps the
    best of overlapping boxes.
13. **Camera compensation end-to-end**: synthetic panning video with one parked and one moving vehicle
    and `FakeDetector` → `run_vehicle_pipeline` reports the parked one STATIONARY (dominant) and the
    moving one MOVING. Run it again with `motion_model="none"` and assert the parked vehicle is then
    wrongly MOVING (proves compensation matters, as in the reference project's ablation).
14. Pipeline output: MP4 exists, `probe_video` on it gives the same duration as the source within
    one frame and the render size; JSON has `summary.vehicles.unique_track_ids == 2`. Skip with
    `pytest.mark.skipif` when no ffmpeg can be found (§4.7), so CI without ffmpeg still passes the rest.
15. `render.draw_frame` returns the same shape, does not modify the input, and the panel area is darker.

Endpoint tests (Django `TestCase` + `APIClient`, patching seams like `tests/test_corners.py` does):
16. 401 without auth; 400 missing `account_id`; 400 missing `video_id`; 404 for another account's video.
17. `wait=true` success: patch `core.azure.blob.download_blob_to_temp`,
    `core.vehicles.pipeline.run_vehicle_pipeline` (writes a tiny fake MP4 + returns a report),
    `core.azure.blob.put_blob_and_sas` (returns fake SAS URLs), and `find_spec` to pretend ultralytics
    exists → 200 with `downloadUrl` ending in the expected blob name, and `put_blob_and_sas` called with
    `content_type="video/mp4"` and blob `"{account}/images/{pk}/vehicles/vehicles_tracked.mp4"`.
18. `wait=true` with ultralytics missing → 503, `unavailable: true`, and the video was **not** downloaded.
19. `wait=true` with the pipeline raising `RuntimeError` → 500, `unavailable: false`, temp files removed.
20. Async: PUT returns 202 `queued|running`; patch `threading.Thread` to run synchronously or join it;
    then GET returns `completed` once the blob-exists check is patched true.
21. Existing result and no `force` → 200 immediately without running the pipeline; `force=true` reruns.
22. `start_s`/`end_s` validation (non-numeric → 400, `end_s <= start_s` → 400) and they reach `build_config`.

Use the Django test cache (LocMem). Clear `vehicle_jobs` cache keys and `_active` in `setUp`.

---

## 8. Live verification (after unit tests pass)

1. Install the optional stack into `~/dvsa-venv`: `pip install -r requirements/vehicles.txt` plus the
   CUDA 12.8 torch wheels (the dev box has an RTX 5060 Laptop GPU; check `torch.cuda.is_available()`).
2. Pick a clip **with vehicles** (Q3). Run the CLI on 20 s of it:
   `~/dvsa-venv/Scripts/python -m core.vehicles <clip.mp4> --out-dir %TEMP%\vehicles --end 20`.
   Open `vehicles_tracked.mp4` in a browser tab (proves H.264 + faststart). Check visually: boxes on
   vehicles, stable IDs, parked vehicles cyan while the drone pans, moving vehicles orange with trails.
3. Record timings from the summary in the PR description (detection ms/frame, total, realtime factor).
4. Endpoint: start the API with the env from `local-startup-scripts/local.ps1` (do not print it),
   `runserver 127.0.0.1:8000 --noreload`, get a JWT for the test user, then
   `PUT /api/v1/videos/vehicles/ {"account_id": 5, "video_id": <pk>, "wait": true, "end_s": 20}` and
   open the returned `downloadUrl`. Also try the async path with GET polling.
5. Negative live check: run with ultralytics uninstalled in a scratch venv or `DVSA_VEHICLES_WEIGHTS`
   set to a bad name → 503 and the rest of the API still responds.

---

## 9. Stages and commits (one commit each; run the listed tests before committing)

| # | Stage | Files | Done when |
|---|---|---|---|
| S0 | Plan (this doc) | `docs/vehicle_tracking_plan.md` | User confirms the plan and answers Q1–Q5 (or accepts defaults). |
| S1 | Package skeleton, video I/O, ffmpeg discovery | `core/vehicles/__init__.py`, `video_io.py`, `ffmpeg.py`, `requirements/base.txt` (`imageio-ffmpeg`) | tests 1, 2 pass. |
| S2 | Camera motion | `core/vehicles/motion.py` | tests 3–5 pass. |
| S3 | Tracker | `core/vehicles/tracker.py`, tiling helpers `core/vehicles/boxes.py` | tests 6–10, 12 pass. |
| S4 | Activity | `core/vehicles/activity.py` | test 11 passes. |
| S5 | Detector interface + Ultralytics impl | `core/vehicles/detector.py`, `requirements/vehicles.txt`, `.gitignore` | module imports without ultralytics installed; `VehiclesUnavailable` raised lazily. |
| S6 | Render | `core/vehicles/render.py` | test 15 passes. |
| S7 | Pipeline + CLI | `core/vehicles/pipeline.py`, `core/vehicles/__main__.py`, `tests/vehicle_fixtures.py` | tests 13–14 pass. |
| S8 | Job, view, URL, settings | `apps/videos/vehicle_jobs.py`, `apps/videos/views.py`, `apps/videos/urls.py`, `config/settings/base.py` | tests 16–22 pass; full suite passes. |
| S9 | Live verification | none (results in PR text) | §8 steps 2–5 done; MP4 plays in the browser. |
| S10 | Push + PR (only after the user says so) | — | PR "feat(videos): vehicle tracking video endpoint (/videos/vehicles/)" with summary, API contract, timings, licence note, limitations. |

---

## 10. Out of scope (do not build unless asked)

- dvsa-platform UI button (can follow the orthomosaic-button pattern later).
- Automatic run on upload (a `post_save` signal); the orthomosaic PR shows how if wanted later.
- Metric speeds (needs ground-sample distance from altitude/camera intrinsics or SRT telemetry).
- Appearance re-ID across exits, offline tracklet stitching, line/zone counting, licence-plate reading.
- Fine-tuning a detector.

---

## 11. Open questions (defaults in bold are what the implementer uses if the user does not answer)

- **Q1 Detector licence.** Ultralytics + official weights are AGPL-3.0. Options: **(a) accept AGPL for
  now (default; matches the reference project)**; (b) Apache-2.0 alternative such as RT-DETR via Hugging
  Face `transformers` or YOLOX; (c) the repo's existing `custom_models` ONNX adapter with a model the
  owner has rights to. The `VehicleDetector` interface keeps this swappable.
- **Q2 Aerial domain.** COCO models miss small top-down vehicles from high altitude. **Default: COCO
  weights, `imgsz=1280`, tiling off**; enable `DVSA_VEHICLES_TILE=true` for 4K/high-altitude footage.
  Later options: VisDrone-trained weights (check their licence) or YOLOE visual prompts as in the
  reference project.
- **Q3 Test footage.** Which uploaded video (account 5) contains vehicles? **Default: the user names a
  `video_id`; if none, use a public-domain road clip locally via the CLI only** (never commit video files).
- **Q4 Execution model.** **Default: in-process daemon thread + cache state + `wait=true` option**
  (same as PR #32). Production alternative: a Celery task (celery/redis already in requirements).
- **Q5 Output retention.** **Default: overwrite the same blob on `force`, SAS valid 24 h, no cleanup.**
