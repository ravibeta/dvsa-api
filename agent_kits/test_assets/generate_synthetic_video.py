"""Generate a small, deterministic synthetic drone video for tests & demos.

Primary output is a ``.json`` *video manifest* (frames + synthetic ground-truth
objects) that the offline pipeline consumes directly — no codecs required, fully
reproducible under a fixed seed. When OpenCV is installed and ``--mp4`` is passed a
tiny companion ``.mp4`` is also written; CI relies only on the JSON manifest.

Usage::

    python agent_kits/test_assets/generate_synthetic_video.py --out sample_short.json
    python agent_kits/test_assets/generate_synthetic_video.py --out sample.json --mp4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

DEFAULT_SEED = 1234
DEFAULT_FRAMES = 15
DEFAULT_FPS = 5
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
_CLASSES = ["car", "person", "truck"]


def build_manifest(
    *,
    frames: int = DEFAULT_FRAMES,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Build a deterministic video manifest dict (same seed → identical output)."""
    rng = random.Random(seed)
    frame_list: List[Dict[str, Any]] = []
    # A couple of persistent tracks that drift across frames + occasional extras.
    tracks = [
        {"class_name": "car", "x": 40.0, "y": 60.0, "dx": 12.0, "dy": 2.0},
        {"class_name": "person", "x": 300.0, "y": 220.0, "dx": -3.0, "dy": 4.0},
    ]
    for i in range(frames):
        objects: List[Dict[str, Any]] = []
        for track in tracks:
            x = track["x"] + track["dx"] * i
            y = track["y"] + track["dy"] * i
            objects.append({
                "bbox": [round(x % width, 2), round(y % height, 2), 40.0, 30.0],
                "class_name": track["class_name"],
                "confidence": round(0.80 + rng.random() * 0.15, 4),
            })
        # Deterministic sprinkling of a transient detection.
        if i % 5 == 2:
            objects.append({
                "bbox": [round(rng.random() * width, 2), round(rng.random() * height, 2),
                         50.0, 35.0],
                "class_name": _CLASSES[i % len(_CLASSES)],
                "confidence": round(0.55 + rng.random() * 0.2, 4),
            })
        frame_list.append({
            "index": i,
            "timestamp": round(i / fps, 6),
            "width": width,
            "height": height,
            "objects": objects,
        })
    return {
        "schema": "dvsa-synthetic-video/1",
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
        "frames": frame_list,
    }


def ensure_manifest(path: str, **kwargs: Any) -> str:
    """Write the manifest to ``path`` if missing; return the path. CI fallback."""
    if not os.path.exists(path):
        write_manifest(path, **kwargs)
    return path


def write_manifest(path: str, *, mp4: bool = False, **kwargs: Any) -> str:
    manifest = build_manifest(**kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    if mp4:
        _maybe_write_mp4(os.path.splitext(path)[0] + ".mp4", manifest)
    return path


def _maybe_write_mp4(path: str, manifest: Dict[str, Any]) -> None:
    """Write a tiny companion MP4 if OpenCV+NumPy are available (best-effort)."""
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - MP4 is optional; JSON manifest is canonical
        print("opencv/numpy not available — skipping .mp4 (JSON manifest is enough)")
        return
    w, h, fps = manifest["width"], manifest["height"], manifest["fps"]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in manifest["frames"]:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for obj in frame["objects"]:
            x, y, bw, bh = (int(v) for v in obj["bbox"])
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        writer.write(img)
    writer.release()
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic drone video")
    parser.add_argument("--out", default="sample_short.json", help="output path")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mp4", action="store_true", help="also write .mp4 if OpenCV present")
    args = parser.parse_args()
    path = write_manifest(args.out, mp4=args.mp4, frames=args.frames, fps=args.fps,
                          seed=args.seed)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
