# Kit 1 — Local Interactive

**Purpose.** Give a developer or operator an agent-guided way to run and experiment
with the drone-video pipeline locally — one skill at a time, with safe defaults and
explicit confirmation before anything destructive.

## Architecture

```
 cli_wrapper.py ──▶ agent_harness.LocalAgentHarness ──▶ agent_kits.common.run_pipeline
        │                    │                                    │
        │             agent_profile.yaml                   fetch → extract →
        │             (autonomy + gating)                  infer → summarise
        └── skills/*.skill  ──(dynamic load)──▶ steps: extract_frames /
                                                 detect_objects / summarize_timeline
```

The harness loads the profile and every `*.skill` file at runtime, enforces the
autonomy mode, and gates *always-ask* actions (like `write_output`).

## Quickstart

```bash
# Generate the deterministic synthetic video (JSON manifest; no codecs needed)
python agent_kits/test_assets/generate_synthetic_video.py \
    --out agent_kits/test_assets/sample_short.json

VIDEO=agent_kits/test_assets/sample_short.json

# Dry-run: describe the plan, execute nothing
python -m agent_kits.local_interactive.cli_wrapper --dry-run run --video $VIDEO --fps 5

# Individual steps
python -m agent_kits.local_interactive.cli_wrapper extract-frames --video $VIDEO --fps 5
python -m agent_kits.local_interactive.cli_wrapper run-inference  --video $VIDEO --fps 5

# Full run + persist output (write-output is always-ask; --yes auto-confirms)
python -m agent_kits.local_interactive.cli_wrapper --yes run --video $VIDEO --fps 5 --out ./out
```

CLI commands: `fetch-video`, `extract-frames`, `run-inference`, `write-output`,
`run`. Logs are structured JSON on **stderr**; the command result is JSON on
**stdout**.

## Autonomy & gating

`agent_profile.yaml` declares:

- **`autonomy.default_mode`** — `dry_run` (describe only), `confirm` (run read-only
  steps, prompt before gated actions), or `auto` (run everything permitted).
- **`autonomy.max_mode`** — a ceiling the harness never exceeds.
- **`always_ask`** — actions that always require explicit confirmation
  (`write_output`, `delete_artifacts`, `publish_results`).
- **`guardrails`** — `max_fps`, `max_frames`, `require_local_paths` (cost/safety).

In code:

```python
from agent_kits.local_interactive.agent_harness import LocalAgentHarness
from agent_kits.common import RunInput

harness = LocalAgentHarness(mode="confirm", confirm_fn=lambda action, detail: True)
result = harness.run(RunInput(video_uri="file://.../sample_short.json",
                              processing_flags={"fps": 5}), store_dest="./out")
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `AGENT_KITS_LOG_LEVEL` | Log verbosity (default `INFO`). |

(The local kit is offline-first and needs no credentials.)

## Sample output

```json
{"executed": true,
 "output": {"run_id": "run_…", "detections": [...],
            "summary": {"total_detections": 33, "by_class": {"car": 16, "person": 16},
                        "frames_processed": 15}},
 "stored_uri": "file:///…/run_….json"}
```

## Cost control

- Always `--dry-run` first to validate inputs before spending inference.
- Bound `--fps`; the profile's `max_fps` guardrail rejects runaway rates.
- Use single-step commands (`extract-frames`) to iterate cheaply.

## Extension guide

- **Add a skill**: drop a new `name.skill` YAML into `skills/` with a `step:` the
  harness understands (or extend `agent_harness.run_skill`). It is discovered
  automatically — no registration.
- **Swap inference/storage**: pass a custom `InferenceAdapter`/`StorageAdapter` (see
  `agent_kits/README.md` → Extension guide).

## Model-version pinning

Pass `--model-version yolo-v8@2026-07` (or set it on your `InferenceAdapter`) so
every detection records exactly which model produced it. Avoid floating tags.
