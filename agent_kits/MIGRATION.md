# Migration Guide — Adopting the Agent Kits

The agent kits are **additive**: they live entirely under `agent_kits/`, introduce no
changes to existing DVSA API modules, and are opt-in. There is nothing to migrate to
keep current functionality working. This guide covers adopting the kits and moving
between the offline defaults and production infrastructure.

## From nothing → running a kit

1. Install the kit dependencies (or rely on the built-in fallbacks):
   ```bash
   pip install -r agent_kits/requirements.txt
   ```
   Without them, the package still imports (Pydantic shim + minimal YAML loader) and
   the offline pipeline runs; you only lose the FastAPI runner and full validation.
2. Generate the deterministic test asset:
   ```bash
   python agent_kits/test_assets/generate_synthetic_video.py \
     --out agent_kits/test_assets/sample_short.json
   ```
3. Run any kit (see each kit's README quickstart).

## From offline defaults → production adapters

The offline reference adapters are swap-in points, not fixtures:

| Concern | Offline default | Production swap |
| --- | --- | --- |
| Video fetch | `LocalVideoFetcher` | S3/GCS/HTTP `VideoFetcher` |
| Frame extract | `SyntheticFrameExtractor` | OpenCV / decoder `FrameExtractor` |
| Inference | `MockInferenceAdapter` | Triton/YOLO/ONNX `InferenceAdapter` |
| Storage | `LocalFileStorageAdapter` | Object-store `StorageAdapter` |
| Dispatch (triggers) | in-process `make_default_dispatcher` | cloud-runner / queue dispatcher |

Each is a `Protocol` in `agent_kits/common/adapters.py`; implement the interface and
pass your instance into `run_pipeline`, `execute_run`, or the orchestrator's
`executor`. No core edits required.

## From the mock model → a pinned real model

Set a fully-qualified `model_version` on your `InferenceAdapter` (e.g.
`yolo-v8@2026-07-01`). It flows into every `Detection` and the `RunOutput`, so
historical runs remain attributable after you upgrade models.

## Versioning & compatibility

- The kit version lives in `agent_kits/VERSION` (currently `0.1.0`) and follows
  SemVer. Breaking changes to `RunInput`/`Detection`/`RunOutput` will bump the major
  version and be called out in `CHANGELOG.md`.
- The schemas are exported programmatically (`export_schemas()`); pin against them if
  you build external clients.

## Test asset note

The spec references `sample_short.mp4`; this implementation ships a deterministic
`sample_short.json` *video manifest* instead so tests need no video codecs and stay
byte-reproducible. A companion `.mp4` can be generated with `--mp4` when OpenCV is
installed. This is the only intentional deviation from the original asset list.
