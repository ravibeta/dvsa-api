# Kit 3 — Orchestration (Parent/Child)

**Purpose.** Scale processing by splitting a video into partitions, executing them
in parallel (locally or via the cloud runner), and merging the results into a single
canonical, deterministic timeline.

## Architecture

```
              ParentOrchestrator
                    │
      build_partitions (time windows × spatial tiles, with overlap)
                    │
        ┌───────────┼───────────┐          executor(partition) → RunOutput
        ▼           ▼           ▼           (in-process child_worker by default,
   child w0_t0  child w1_t0  child w2_t0     or a cloud-runner / local-queue executor)
        └───────────┼───────────┘
                    ▼
        merge_run_outputs  (IoU overlap removal + confidence conflict resolution)
                    ▼
        canonical timeline  ──▶  optional critic_review (human sample / secondary model)
```

## Quickstart

```bash
VIDEO=agent_kits/test_assets/sample_short.json

# Partition into 3 time windows with 10% overlap, merge with IoU 0.5
python -m agent_kits.orchestration.parent_orchestrator \
    --video $VIDEO --windows 3 --overlap 0.1 --iou 0.5 --out ./merged.json

# Add a 2x2 spatial tiling as well
python -m agent_kits.orchestration.parent_orchestrator \
    --video $VIDEO --windows 2 --tiles-x 2 --tiles-y 2 --overlap 0.15
```

In code:

```python
from agent_kits.orchestration.parent_orchestrator import ParentOrchestrator
from agent_kits.common import RunInput

merged = ParentOrchestrator().run(
    RunInput(video_uri="file://…/sample_short.json", processing_flags={"fps": 5}),
    windows=3, tiles=(2, 2), overlap=0.2, iou_threshold=0.5)
```

### Custom executor (cloud / queue)

The executor is any `Callable[[RunInput], RunOutput]`:

```python
def cloud_executor(partition):
    resp = requests.post(f"{RUNNER_URL}/run", json=partition.model_dump())
    return RunOutput.model_validate(resp.json())

ParentOrchestrator(executor=cloud_executor).run(run_input, windows=8)
```

## Determinism & merging

- **Partitioning** is purely a function of span/grid/overlap — reproducible.
- **Merging** removes overlaps by IoU at matching timestamps and resolves conflicts
  by keeping the highest-confidence detection.
- The merged **detection data** is byte-for-byte identical across runs — use
  `canonical_timeline(merged)` for the provenance-free reproducible view. Individual
  detections still carry per-run `trace_id` lineage.

## Critic / review layer (optional)

```python
from agent_kits.orchestration.critic_review import review
report = review(run_input, merged, human_sample_fraction=0.1, run_secondary=True)
```

- **Human review sampling** — deterministic, seeded selection of a fraction of
  detections to spot-check.
- **Secondary-model validation** — re-runs a *different* `InferenceAdapter` and
  reports agreement plus primary-only/secondary-only detections.

It never mutates the run output; it only emits a report.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `RUNNER_MAX_PARALLELISM` | Suggested cap when wiring a concurrent executor. |
| `AGENT_KITS_LOG_LEVEL` | Log verbosity. |

## Sample output

```json
{"merged": {"run_id": "merge_…", "detections": [...],
            "summary": {"total_detections": 33, "partitions_merged": 3,
                        "pre_merge_detections": 41, "child_run_ids": ["base-w0_t0", ...]}},
 "timeline": [{"timestamp": 0.0, "class_name": "car", "confidence": 0.92,
               "bbox": [40,60,40,30]}]}
```

## Cost control

- More `--windows`/tiles + higher `--overlap` improves border recall but costs more
  inference; tune overlap down to save cost.
- Run the critic on a small `human_sample_fraction`; enable `run_secondary` only for
  audits, not every run.

## Extension guide

- **Partition strategy**: add new `partition_*` functions and wire them into
  `build_partitions`.
- **Executor**: supply a cloud/queue executor to distribute children.

## Model-version pinning

Children stamp their `model_version` onto every detection; the merge records the set
of model versions used so a mixed-model merge is auditable. Pin each child's adapter
to a fully-qualified tag.
