# `custom_models/` — pluggable multi-format model selection

A small layer that lets **dvsa-api** *choose* among several curated drone /
aerial object-detection models and run any of them behind **one** detector
interface. It supports **ONNX**, **PyTorch (TorchScript / `.pt`)** and
**Ultralytics YOLO** artifacts, plus an **Azure Custom Vision** export helper.

> This package complements the existing single-format `custom_model/` (singular)
> ONNX package — it does **not** replace it. The ONNX adapter here *delegates* to
> the proven `custom_model.onnx_inference.CustomONNXDetector`, so behaviour and
> the `(x, y, w, h)` output contract are identical.

## The common detector interface

Every adapter implements the same three methods and returns the same dict shape
as `apps.analytics.routines.base.Detection`:

```python
detector.load()                       # -> self (build session/model, load labels)
detector.infer(frame)                 # frame: BGR ndarray (OpenCV)
# -> [{"label": "vehicle", "score": 0.92, "bbox": [x, y, w, h]}, ...]
detector.close()                      # release the runtime
```

`bbox` is `(x, y, w, h)` — top-left + width/height — in **original-frame pixels**.

## Files

| File | Purpose |
| --- | --- |
| `loader.py` | `ModelSpec` dataclass, `discover_model(path)`, `load_label_map(path)`. |
| `postprocess.py` | Shared box-scaling → `(x,y,w,h)` + NMS (re-uses `custom_model.onnx_inference`). |
| `adapters/onnx_adapter.py` | `ONNXAdapter` — delegates to the existing ONNX detector. |
| `adapters/torch_adapter.py` | `TorchAdapter` — TorchScript/`.pt` + SAHI-style tiling. |
| `adapters/yolo_adapter.py` | `YOLOAdapter` — Ultralytics `.pt`/ONNX, auto-detected. |
| `azure/customvision_adapter.py` | Export an Azure Custom Vision iteration to ONNX → `ModelSpec`. |
| `registry.py` | `register` adapters; `get_detector(spec_or_id)`. |
| `selector.py` | `ModelSelector` — query (task/classes/altitude/resolution) → best `ModelSpec`. |
| `models_catalog.json` | Curated model catalog (VisDrone, TPH-YOLOv5, DOTA, Ultralytics …). |
| `label_map.json.example` | Example VisDrone `{class_id: label}` map. |
| `../tests/test_loader.py` `../tests/test_adapters.py` `../tests/test_selector.py` | Tests (mock all runtimes). |

## Quick start

### Pick a model with the selector, then run it

```python
from custom_models import ModelSelector, get_detector
import cv2

selector = ModelSelector.default()                       # loads models_catalog.json
spec = selector.select(
    task="detection",
    classes=["vehicle", "person"],
    altitude="high",
    resolution=(3840, 2160),                             # 4K → prefers a tiling model
)
spec.path = "/weights/tph-yolov5.pt"                     # point at your local weights
spec.labels_file = "/weights/visdrone.json"

detector = get_detector(spec).load()
frame = cv2.imread("aerial.jpg")                         # BGR
detections = detector.infer(frame)
detector.close()
```

### Or address a model directly by catalog id / path

```python
from custom_models import get_detector, discover_model

# By catalog id (resolved from the bundled models_catalog.json):
detector = get_detector("ultralytics-yolov8-coco")

# Or discover the format from a file path:
spec = discover_model("/weights/yolov8x.pt", labels_file=None)  # -> format "yolo"
detector = get_detector(spec).load()
```

## Curated catalog

`models_catalog.json` ships **metadata only** — no weights. Download each model
from its `source_url`, drop the file at `artifact_filename`, then either set
`ModelSpec.path`/`labels_file` to the local paths or pass `base_dir=` when
loading the catalog (`ModelSelector.from_file(path, base_dir="/weights")`).

| id | format | source |
| --- | --- | --- |
| `visdrone-yolov8x` | yolo | huggingface.co/dronefreak/visdrone-yolov8x |
| `tph-yolov5` | yolo | github.com/cv516Buaa/tph-yolov5 |
| `dota-faster-rcnn` | torch | github.com/jessemelpolio/Faster_RCNN_for_DOTA |
| `visdrone-toolkit-yolov5` | yolo | github.com/dronefreak/VisDrone-dataset-python-toolkit |
| `ultralytics-yolov8-coco` | yolo | github.com/ultralytics/ultralytics |

## Azure Custom Vision (optional)

```python
from custom_models.azure import CustomVisionAdapter, spec_from_export
from custom_models import get_detector

cv = CustomVisionAdapter(project_id, iteration_id, training_key, endpoint)
download_uri = cv.export_onnx()           # async export → download URI (no keys hard-coded)
# ... download + unzip to model.onnx / label_map.json ...
spec = spec_from_export("model.onnx", labels_file="label_map.json")
detector = get_detector(spec).load()
```

## Tests / CI

```bash
pip install -r custom_models/requirements.txt
pytest tests/test_loader.py tests/test_adapters.py tests/test_selector.py -q
```

* The tests **mock the runtimes** — `onnxruntime` via an injected `session_factory`,
  `torch.jit.load` via an injected `model_loader`, and `ultralytics.YOLO` via an
  injected `model_factory` — so **no real model binaries or GPU are required**.
  Only `numpy` and `opencv-python` are needed.
* To run without the repo's Django/coverage `addopts`:

  ```bash
  pytest tests/test_loader.py tests/test_adapters.py tests/test_selector.py -o addopts="" -q
  ```

## Adding a new format

```python
from custom_models.registry import register

@register("myformat")
class MyAdapter:
    format = "myformat"
    def __init__(self, spec, **kwargs): ...
    def load(self): ...; return self
    def infer(self, frame): ...        # -> [{"label","score","bbox":[x,y,w,h]}]
    def close(self): ...
```

Then any `ModelSpec(format="myformat", ...)` (or catalog entry) is runnable via
`get_detector(spec)`.
