# Custom ONNX model adapter (`custom_model/`)

A pluggable adapter to run **custom ONNX object-detection models** exported from
[LandingLens](https://landing.ai/) (or re-exported via Azure Custom Vision)
inside **dvsa-api**. It exposes a small `CustomONNXDetector` with `load()`,
`infer(frame)` and `close()` methods. Use the factory `create_detector(config)`
to build one from a `ModelConfig`.

The runtime ONNX session is **injectable**, so the whole pipeline is unit-tested
without a real `.onnx` file or even the `onnxruntime` package installed (the
tests mock `onnxruntime.InferenceSession`).

## Files

| File | Purpose |
| --- | --- |
| `model_loader.py` | `ModelConfig` dataclass, `load_label_map`, `validate_model_file`, `create_detector` factory. |
| `onnx_inference.py` | `CustomONNXDetector` — preprocessing, inference, postprocessing, tiling + NMS. |
| `azure_customvision_helper.py` | Optional helper to export an ONNX iteration from Azure Custom Vision. |
| `label_map.json.example` | Example `{class_id: label}` map — copy to `label_map.json`. |
| `requirements.txt` | Dependencies for this module. |
| `../tests/test_custom_model_integration.py` | Full-pipeline tests with a mocked session. |
| `../tests/test_onnx_inference_preprocessing.py` | Preprocessing/tiling unit tests. |

## Expected model artifact

```
/path/to/custom_model.onnx      # the exported model
/path/to/label_map.json         # {"0":"person","1":"vehicle","2":"bicycle"}
```

The detector handles the three most common ONNX detection output layouts:

1. Three arrays — `[boxes(N,4), scores(N,), labels(N,)]`.
2. A single combined array — `(N, 6)` rows of `[x1, y1, x2, y2, score, class_id]`.
3. A `dict` keyed by `boxes`/`scores`/`labels` (TF-style `detection_*` keys also work).

Boxes may be **normalised** `[0, 1]` or expressed in **input-pixel** space; the
adapter detects which and returns `bbox` as `(x, y, w, h)` in **original frame
pixel coordinates** either way. This matches
`apps.analytics.routines.base.Detection`, so detections drop straight into the
analytics pipeline. (Models almost always emit corner boxes `[x1, y1, x2, y2]`;
the adapter converts on the way out — `xyxy_to_xywh` / `xywh_to_xyxy` are
exported from `onnx_inference` for callers that need the other form.)

## Quick start (local smoke test)

```bash
pip install -r custom_model/requirements.txt
```

```python
from custom_model.model_loader import ModelConfig, create_detector
import cv2

config = ModelConfig(
    onnx_path="/path/to/custom_model.onnx",
    labels_path="/path/to/label_map.json",
    input_size=(640, 640),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    tile_size=None,            # set e.g. (1024, 1024) for large aerial frames
)

detector = create_detector(config)
detector.load()

frame = cv2.imread("sample_frame.jpg")        # BGR, as OpenCV returns
detections = detector.infer(frame)
# detections -> [{"label": "vehicle", "score": 0.92, "bbox": [x, y, w, h]}, ...]
# bbox is (x, y, w, h) in original-frame pixels — same as base.Detection.

detector.close()
```

### Converting a LandingLens `labels.txt` to `label_map.json`

LandingLens exports one class name per line (the line number is the class id):

```
person
vehicle
bicycle
```

```python
import json

with open("labels.txt") as fh:
    labels = [line.strip() for line in fh if line.strip()]

label_map = {str(i): name for i, name in enumerate(labels)}
json.dump(label_map, open("label_map.json", "w"), indent=2)
# -> {"0": "person", "1": "vehicle", "2": "bicycle"}
```

## Integrate into `dvsa-api`

The wiring is **already implemented** in
`apps/analytics/routines/custom_onnx.py` and registered in the routine registry
(`apps/analytics/routines/__init__.py`). Because the adapter returns
`(x, y, w, h)` boxes, detections map cleanly onto `base.Detection`.

### 1. Configuration (env)

Set these (see `.env.example`). When unset, the routine stays registered but is
inert — existing analytics are unaffected.

```env
CUSTOM_MODEL_ONNX_PATH=/secrets/custom_model.onnx
CUSTOM_MODEL_LABELS_PATH=/secrets/label_map.json
# Optional:
# CUSTOM_MODEL_INPUT_SIZE=640x640
# CUSTOM_MODEL_MEAN=0.485,0.456,0.406
# CUSTOM_MODEL_STD=0.229,0.224,0.225
# CUSTOM_MODEL_TILE_SIZE=1024x1024
# CUSTOM_MODEL_TILE_OVERLAP=0.2
# CUSTOM_MODEL_SCORE_THRESHOLD=0.25
```

### 2. Run it through the existing analytics pipeline

The detector is exposed as the frame-level routine **`custom_onnx_detection`**,
so it works with the existing `RunAnalysisView` / `run_video_analysis` Celery
flow and the `RoutineListView` discovery endpoint — no new endpoint required:

```jsonc
// POST /api/analytics/videos/<video_id>/run
{ "routines": ["custom_onnx_detection"], "frame_step": 30, "max_frames": 300 }
```

It returns the standard envelope, with `bbox` as `(x, y, w, h)`:

```json
{
  "routine": "custom_onnx_detection",
  "summary": {"count": 1, "labels": ["vehicle"]},
  "detections": [{"bbox": [120, 80, 480, 320], "centroid": [360.0, 240.0],
                  "area": 153600.0, "label": "vehicle", "score": 0.92}]
}
```

Programmatic use is identical to any other routine:

```python
from apps.analytics.routines import run_frame_routine

result = run_frame_routine("custom_onnx_detection", frame)  # frame: BGR ndarray
```

The detector is a process-wide singleton (`get_custom_detector()`, cached);
call `reset_custom_detector()` after changing the environment.

### 3. Optional: a dedicated single-frame endpoint

If you want a synchronous image endpoint in addition to the video pipeline:

```python
from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np
import cv2

from apps.analytics.routines.custom_onnx import get_custom_detector

router = APIRouter()


@router.post("/detect/custom")
async def detect_custom(file: UploadFile = File(...)):
    detector = get_custom_detector()
    if detector is None:
        raise HTTPException(503, "Custom ONNX model is not configured")

    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    return {"detections": detector.infer(frame)}  # bbox = [x, y, w, h]
```

A Django REST Framework equivalent is an `APIView.post` that reads
`request.FILES["file"].read()` and calls the same `detector.infer(frame)`.

## Preprocessing & tiling

For high-resolution aerial frames, set `ModelConfig.tile_size` (and optionally
`tile_overlap`, a fraction in `[0, 1)`). The detector splits the frame into
overlapping tiles, runs inference per tile, maps detections back to absolute
coordinates and merges duplicates with a simple per-label IoU **NMS**
(`iou_threshold`). Swap `_nms` for your repo's NMS if you have a tuned one.

## Azure Custom Vision (optional)

`azure_customvision_helper.py` is a minimal, secret-free `requests` wrapper to
**export** a trained Custom Vision iteration as ONNX. For first-time setup the
Azure portal or the official SDK is recommended. No keys are hard-coded — pass
them in from a secret store.

## Tests / CI notes

```bash
pip install -r custom_model/requirements.txt
pytest tests/test_custom_model_integration.py tests/test_onnx_inference_preprocessing.py -q
```

* The tests **mock `onnxruntime.InferenceSession`** via the injected
  `session_factory`, so `onnxruntime` is **not** required to run them — only
  `numpy` and `opencv-python` (for resize/colour-convert).
* They do not require Django. If your CI runs the whole suite under the repo's
  `pytest.ini` (which sets `--cov=apps` and a Django settings module), these
  tests still pass; they live in `tests/` like the rest of the suite.
* To run only this module's tests without the repo's coverage/Django addopts:

  ```bash
  pytest tests/test_custom_model_integration.py tests/test_onnx_inference_preprocessing.py -o addopts="" -q
  ```

## API reference (summary)

| Symbol | Signature |
| --- | --- |
| `ModelConfig` | `(onnx_path, labels_path, input_size=(640,640), mean=(0,0,0), std=(1,1,1), tile_size=None, tile_overlap=0.0, score_threshold=0.0, iou_threshold=0.5)` |
| `create_detector` | `(config, session_factory=None) -> CustomONNXDetector` |
| `load_label_map` | `(path) -> Dict[int, str]` |
| `CustomONNXDetector.load` | `() -> self` |
| `CustomONNXDetector.infer` | `(frame: np.ndarray) -> List[{"label", "score", "bbox":[x,y,w,h]}]` |
| `CustomONNXDetector.close` | `() -> None` |
