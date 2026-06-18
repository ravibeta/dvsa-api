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
adapter detects which and returns `bbox` in **original frame pixel coordinates**
either way.

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
# detections -> [{"label": "vehicle", "score": 0.92, "bbox": [x1, y1, x2, y2]}, ...]

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

dvsa-api already has a pluggable routine registry in
`apps/analytics/routines/base.py`. The custom detector is a *model-based*
detector rather than a classical routine, so the simplest integration is a thin
process-wide singleton built from environment variables.

### 1. Configuration (env)

```env
CUSTOM_MODEL_ONNX_PATH=/secrets/custom_model.onnx
CUSTOM_MODEL_LABELS_PATH=/secrets/label_map.json
```

### 2. A small registry / factory

```python
# apps/analytics/custom_detector.py  (new file — adapt paths to your repo)
import os
from functools import lru_cache
from typing import Optional

from custom_model.model_loader import ModelConfig, create_detector
from custom_model.onnx_inference import CustomONNXDetector


@lru_cache(maxsize=1)
def get_custom_detector() -> Optional[CustomONNXDetector]:
    """Return a loaded detector, or None if not configured."""
    onnx_path = os.getenv("CUSTOM_MODEL_ONNX_PATH")
    labels_path = os.getenv("CUSTOM_MODEL_LABELS_PATH")
    if not onnx_path or not labels_path:
        return None

    cfg = ModelConfig(
        onnx_path=onnx_path,
        labels_path=labels_path,
        input_size=(640, 640),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    detector = create_detector(cfg)
    detector.load()
    return detector
```

> `# TODO` Wire `get_custom_detector()` into your existing detection dispatch so
> a request can route to `"custom_onnx"`. If you prefer the routine registry,
> register a `level="frame"` wrapper that calls `detector.infer(frame)` and maps
> the returned dicts into `apps.analytics.routines.base.Detection` objects
> (note: that `Detection.bbox` is `(x, y, w, h)` whereas this adapter returns
> `[x1, y1, x2, y2]`).

### 3. Example FastAPI endpoint

```python
from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np
import cv2

from apps.analytics.custom_detector import get_custom_detector

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

    return {"detections": detector.infer(frame)}
```

A Django REST Framework equivalent is a `APIView.post` that reads
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
| `CustomONNXDetector.infer` | `(frame: np.ndarray) -> List[{"label", "score", "bbox":[x1,y1,x2,y2]}]` |
| `CustomONNXDetector.close` | `() -> None` |
