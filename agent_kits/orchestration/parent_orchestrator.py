"""Parent orchestrator — partition a video, fan out children, merge deterministically.

Splits a :class:`RunInput` into overlapping partitions (time windows and/or spatial
tiles), executes each via a pluggable *executor* (in-process by default; swap in a
cloud-runner or local-queue executor), then reconciles the child outputs with
:func:`~agent_kits.orchestration.merge_outputs.merge_run_outputs`.

Overlap is configurable to trade recall (more overlap, fewer missed border objects)
against cost. All partitioning is deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..common import (
    RunInput,
    RunLogger,
    RunOutput,
    SyntheticFrameExtractor,
    TimeWindow,
)
from .child_worker_template import process_partition
from .merge_outputs import canonical_timeline, merge_run_outputs

# An executor maps a partition RunInput to a RunOutput (local, cloud, queue, ...).
Executor = Callable[[RunInput], RunOutput]


def partition_time_windows(
    start: float, end: float, count: int, *, overlap: float = 0.0,
) -> List[TimeWindow]:
    """Split ``[start, end]`` into ``count`` overlapping windows.

    ``overlap`` is a fraction (0–0.9) of a window's length added to each side.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    span = max(0.0, end - start)
    width = span / count
    pad = width * overlap
    windows: List[TimeWindow] = []
    for i in range(count):
        w_start = max(start, start + i * width - pad)
        w_end = min(end, start + (i + 1) * width + pad)
        windows.append(TimeWindow(start=str(round(w_start, 6)),
                                  end=str(round(w_end, 6))))
    return windows


def partition_tiles(
    width: int, height: int, nx: int, ny: int, *, overlap: float = 0.0,
) -> List[Dict[str, Any]]:
    """Split a ``width``×``height`` frame into an ``nx``×``ny`` grid of tiles."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    tw, th = width / nx, height / ny
    pad_x, pad_y = tw * overlap, th * overlap
    tiles: List[Dict[str, Any]] = []
    for j in range(ny):
        for i in range(nx):
            x = max(0.0, i * tw - pad_x)
            y = max(0.0, j * th - pad_y)
            x2 = min(float(width), (i + 1) * tw + pad_x)
            y2 = min(float(height), (j + 1) * th + pad_y)
            tiles.append({"x": round(x, 4), "y": round(y, 4),
                          "w": round(x2 - x, 4), "h": round(y2 - y, 4),
                          "col": i, "row": j})
    return tiles


class ParentOrchestrator:
    """Partition → execute → merge, with provenance across the whole run."""

    def __init__(self, *, executor: Optional[Executor] = None) -> None:
        self.executor = executor or process_partition
        self.extractor = SyntheticFrameExtractor()
        self.logger = RunLogger(agent_id="orchestrator")

    def _infer_span(self, run_input: RunInput) -> Tuple[float, float, int, int]:
        """Return (start, end, width, height) by inspecting frame metadata."""
        from ..common import LocalVideoFetcher  # noqa: PLC0415
        path = LocalVideoFetcher().fetch_video(run_input.video_uri)
        frames = self.extractor.extract_frames(path, fps=0)
        if not frames:
            return 0.0, 0.0, 0, 0
        ts = [f.timestamp for f in frames]
        w = max((f.width for f in frames), default=0)
        h = max((f.height for f in frames), default=0)
        return min(ts), max(ts), w, h

    def build_partitions(
        self, run_input: RunInput, *, windows: int = 1,
        tiles: Tuple[int, int] = (1, 1), overlap: float = 0.0,
    ) -> List[RunInput]:
        """Produce the child :class:`RunInput`s for the requested partitioning."""
        start, end, width, height = self._infer_span(run_input)
        time_windows = partition_time_windows(start, end, windows, overlap=overlap) \
            if windows > 1 else [run_input.time_window or TimeWindow()]
        nx, ny = tiles
        spatial = partition_tiles(width or 1, height or 1, nx, ny, overlap=overlap) \
            if (nx > 1 or ny > 1) else [None]

        partitions: List[RunInput] = []
        for wi, tw in enumerate(time_windows):
            for ti, tile in enumerate(spatial):
                flags = dict(run_input.processing_flags)
                partition_id = f"w{wi}_t{ti}"
                flags["partition_id"] = partition_id
                if tile is not None:
                    flags["tile"] = tile
                partitions.append(RunInput(
                    video_uri=run_input.video_uri,
                    sensor_id=run_input.sensor_id,
                    time_window=tw if (tw.start or tw.end) else None,
                    processing_flags=flags,
                    agent_id=f"child-{partition_id}",
                    run_id=f"{run_input.run_id or 'run'}-{partition_id}",
                ))
        return partitions

    def run(
        self, run_input: RunInput, *, windows: int = 1,
        tiles: Tuple[int, int] = (1, 1), overlap: float = 0.0,
        iou_threshold: float = 0.5,
    ) -> RunOutput:
        """Partition, execute all children, and return the merged canonical output."""
        partitions = self.build_partitions(
            run_input, windows=windows, tiles=tiles, overlap=overlap)
        self.logger.info("orchestration_start", partitions=len(partitions),
                         windows=windows, tiles=list(tiles), overlap=overlap)
        child_outputs = [self.executor(p) for p in partitions]
        merged = merge_run_outputs(child_outputs, iou_threshold=iou_threshold)
        self.logger.info("orchestration_done", merged_detections=len(merged.detections),
                         pre_merge=merged.summary.get("pre_merge_detections"))
        return merged


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_kits.orchestration.parent_orchestrator",
        description="Partition a drone video, process children, merge deterministically.")
    parser.add_argument("--video", required=True, help="video URI (file:// or path)")
    parser.add_argument("--windows", type=int, default=1, help="time-window partitions")
    parser.add_argument("--tiles-x", type=int, default=1, help="spatial tile columns")
    parser.add_argument("--tiles-y", type=int, default=1, help="spatial tile rows")
    parser.add_argument("--overlap", type=float, default=0.1, help="overlap fraction")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--iou", type=float, default=0.5, help="merge IoU threshold")
    parser.add_argument("--out", default=None, help="write merged RunOutput JSON here")
    args = parser.parse_args(argv)

    run_input = RunInput(video_uri=args.video, processing_flags={"fps": args.fps},
                         agent_id="orchestrator")
    merged = ParentOrchestrator().run(
        run_input, windows=args.windows, tiles=(args.tiles_x, args.tiles_y),
        overlap=args.overlap, iou_threshold=args.iou)

    payload = {"merged": merged.model_dump(), "timeline": canonical_timeline(merged)}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
