"""Local CLI: ``python -m core.vehicles VIDEO --out-dir DIR [--start S] [--end S] ...``.

Runs the vehicle tracking pipeline on a local file (no Django / Azure) and prints the summary.
"""

from __future__ import annotations

import argparse
import json
import sys

from .detector import DetectorConfig, UltralyticsVehicleDetector
from .pipeline import VehiclePipelineConfig, run_vehicle_pipeline
from .tracker import TrackerConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.vehicles", description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out-dir", default="vehicles_out")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = no cap")
    ap.add_argument("--target-fps", type=float, default=15.0)
    ap.add_argument("--weights", default="yolo26x.pt")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--imgsz", type=int, default=0, help="0 = auto (1.5x frame long side)")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tile", action="store_true")
    ap.add_argument("--motion-model", default="similarity", choices=["similarity", "affine", "homography", "none"])
    args = ap.parse_args(argv)

    cfg = VehiclePipelineConfig(target_fps=args.target_fps, start_s=args.start, end_s=args.end,
                                max_seconds=args.max_seconds, motion_model=args.motion_model,
                                tracker=TrackerConfig(low_thresh=args.conf))
    dcfg = DetectorConfig(weights=args.weights, model_dir=args.model_dir, device=args.device,
                          imgsz=args.imgsz, conf=args.conf, tile=args.tile)

    def progress(done: float, total: float) -> None:
        print(f"  {done:7.1f}s / {total:.1f}s", flush=True)

    report = run_vehicle_pipeline(args.video, args.out_dir, cfg,
                                  lambda frame: UltralyticsVehicleDetector(dcfg, warmup_shape=frame.shape),
                                  progress=progress)
    summary = {k: report[k] for k in ("video", "vehicles", "output", "timing_s", "performance")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
