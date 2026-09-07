# Fabricated-footage screening on upload

DVSA-API accepts **camera-captured drone footage only**. When a video is
uploaded, it is screened for signs of being AI-generated / fabricated *before* it
is stored or registered; a clip that screens as fabricated is refused and the
uploader is told the video must be camera-based. Nothing about the Django models
or the analysis / RAG side changes — this gate lives entirely in the upload path.

## Where it runs

`PUT`/`POST /api/v1/videos/upload-video/` (`VideoUploadAPIView`) runs the screen
as its first step, before the blob upload and `VideoEntity.create_video(...)`. So
a rejected clip is never written to blob storage and never registered — which
also means the ingestion/indexing `post_save` signal never fires for it.

```
upload-video/  →  screen the uploaded file
                    ├─ fabricated  → 400, nothing stored or registered
                    └─ otherwise   → blob upload → VideoEntity.create_video → ingest
```

## The screen

The screening logic lives in `core/azure/synth_screen.py`, ported from the
`drone_synth_screen.py` reference script into a reusable, side-effect-free runner
(the same way `survey_corners.py` was ported). It is tuned for **precision**: the
default verdict is `camera`, and a clip is only flagged when several independent
signs of fabrication agree **and** no positive sign of physical capture is
present. `ffprobe` and `cv2` are imported lazily, so the module import is cheap
and the verdict logic is unit-testable without a video runtime.

| Verdict | Meaning | Upload |
| --- | --- | --- |
| `synthetic-declared` | generator tag or C2PA AI manifest in the container | **rejected** |
| `likely-fabricated` | ≥ 3 fabrication signs, no camera evidence | **rejected** |
| `review` | fewer signs, still no camera evidence | allowed |
| `camera` | positive evidence of physical capture (the default) | allowed |

Only the two clear verdicts (`FABRICATED_VERDICTS`) are rejected; `review` is
ambiguous and is allowed through to avoid blocking legitimate footage.

## Failure behaviour (fail-open)

If the screen cannot run — `ffprobe`/`cv2` unavailable, an unreadable clip, or any
other error — the upload **proceeds** (the verdict falls back to `camera`). A
missing video toolchain therefore never blocks a legitimate upload; the screen
only ever *adds* a rejection for clips it can positively identify as fabricated.

## Testing

`tests/test_synth_screen.py` runs fully offline. It exercises the pure verdict
logic (`decide`) on synthetic feature dicts and patches the `screen` seam at the
view to assert that a fabricated clip is refused with `400` before any blob
upload or registration, that `camera`/`review` footage passes through to upload,
and that a screening failure fails open (the upload still succeeds).
