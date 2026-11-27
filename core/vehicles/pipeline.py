"""Vehicle tracking pipeline: detect -> register camera -> track in the ground plane -> classify -> render.

One sequential pass over the video. Detection runs on frames chosen by timestamp
(:class:`~core.vehicles.video_io.FrameSelector`); every source frame in the processed range is
written to the output MP4 with the latest annotations, so the output keeps the source's duration
and playback speed. Outputs in ``out_dir``: ``vehicles_tracked.mp4`` and ``vehicles_tracking.json``.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np

from .activity import MOVING, STATES, STATIONARY, UNKNOWN, ActivityConfig, TrackActivity
from .detector import VEHICLE_CLASSES, VehicleDetector
from .motion import CameraMotionEstimator, apply_h, transform_boxes
from .render import PanelInfo, VehicleDraw, draw_frame
from .tracker import Track, TrackerConfig, VehicleTracker
from .video_io import FrameSelector, Mp4Writer, even_size, iter_frames, probe_video

VIDEO_NAME = "vehicles_tracked.mp4"
REPORT_NAME = "vehicles_tracking.json"
SCHEMA_VERSION = "1.0"


@dataclass
class VehiclePipelineConfig:
    target_fps: float = 15.0  # detection rate by timestamp; <= 0 = every frame
    start_s: float = 0.0
    end_s: Optional[float] = None
    max_seconds: float = 300.0  # cap on processed duration; <= 0 = no cap
    motion_model: str = "similarity"
    render_scale: float = 1.0
    max_render_edge: int = 1920
    trail_s: float = 2.0
    trajectory_hz: float = 5.0
    ffmpeg: Optional[str] = None
    include_frames: bool = False  # add per-processed-frame counts to the report (debugging)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)


def _render_size(w: int, h: int, cfg: VehiclePipelineConfig):
    scale = cfg.render_scale if cfg.render_scale > 0 else 1.0
    if cfg.max_render_edge and max(w, h) * scale > cfg.max_render_edge:
        scale = cfg.max_render_edge / float(max(w, h))
    return even_size(round(w * scale), round(h * scale))


def _centres(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)


def run_vehicle_pipeline(video_path: str, out_dir: str, cfg: VehiclePipelineConfig,
                         make_detector: Callable[[np.ndarray], VehicleDetector],
                         progress: Optional[Callable[[float, float], None]] = None) -> dict:
    """Process one video; returns the report (also written to ``out_dir/vehicles_tracking.json``)."""
    t_wall0 = time.perf_counter()
    timing: Dict[str, float] = {}

    def lap(key: str, t0: float) -> float:
        now = time.perf_counter()
        timing[key] = timing.get(key, 0.0) + (now - t0)
        return now

    info = probe_video(video_path)
    timing["probe"] = time.perf_counter() - t_wall0
    os.makedirs(out_dir, exist_ok=True)
    video_out = os.path.join(out_dir, VIDEO_NAME)
    json_out = os.path.join(out_dir, REPORT_NAME)

    start = max(0.0, float(cfg.start_s or 0.0))
    end = cfg.end_s
    if cfg.max_seconds and cfg.max_seconds > 0:
        end = start + cfg.max_seconds if end is None else min(end, start + cfg.max_seconds)
    planned = max(0.0, (min(end, info.duration_s) if end is not None else info.duration_s) - start)

    selector = FrameSelector(cfg.target_fps, info.fps_nominal)
    gmc = CameraMotionEstimator(cfg.motion_model)
    tracker = VehicleTracker(cfg.tracker)
    activities: Dict[int, TrackActivity] = {}
    state_log: Dict[int, List[tuple]] = {}  # uid -> [(t, state, speed_bl_s)]
    motion_ok_log: deque = deque()
    frames_out: List[dict] = []
    proc_times: List[float] = []

    detector = None
    writer: Optional[Mp4Writer] = None
    pending = None
    items: List[VehicleDraw] = []
    panel: Optional[PanelInfo] = None
    t_first = None
    last_t = 0.0
    n_decoded = 0
    camera_failures = 0
    peak = (0, None)
    next_progress = 0.0

    t0 = time.perf_counter()
    try:
        for idx, t, frame in iter_frames(video_path, start):
            t0 = lap("decode", t0)
            if t < start - 1e-9:
                continue
            if end is not None and t > end + 1e-9:
                break
            n_decoded += 1
            h, w = frame.shape[:2]
            if t_first is None:
                t_first = t
                t0 = time.perf_counter()
                detector = make_detector(frame)
                t0 = lap("model_init", t0)
                writer = Mp4Writer(video_out, *_render_size(w, h, cfg), info.fps_nominal, ffmpeg=cfg.ffmpeg)

            if selector(t):
                boxes, scores, classes = detector.detect(frame)
                boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
                scores = np.asarray(scores, dtype=np.float64).reshape(-1)
                classes = list(classes)
                t0 = lap("detection", t0)

                step = gmc.update(frame, boxes[scores >= cfg.tracker.low_thresh])
                H = step.H_cur_to_ref
                H_inv = np.linalg.inv(H)
                boxes_ref = transform_boxes(H, boxes)
                camera_failures += 0 if step.ok else 1
                motion_ok_log.append((t, step.ok))
                while motion_ok_log and motion_ok_log[0][0] < t - cfg.activity.window_s - 1e-9:
                    motion_ok_log.popleft()
                t0 = lap("motion", t0)

                def in_view(tr: Track, H_inv=H_inv, w=w, h=h) -> bool:
                    cx, cy = apply_h(H_inv, tr.kf.x[None, :2])[0]
                    return -5 <= cx <= w + 5 and -5 <= cy <= h + 5

                matched = tracker.update(t, idx, boxes_ref, boxes, scores, classes, in_view=in_view)
                t0 = lap("tracking", t0)

                window_ok = all(ok for _, ok in motion_ok_log)
                items = []
                by_state = {s: 0 for s in STATES}
                by_class: Dict[str, int] = {}
                for tr, j in matched:
                    act = activities.setdefault(tr.uid, TrackActivity(cfg.activity))
                    recent = [o for o in tr.obs if o.t > t - cfg.activity.window_s - 1e-9]
                    br = np.array([o.box_ref for o in recent])
                    state = act.update(t, [o.t for o in recent], _centres(br),
                                       np.maximum(br[:, 2] - br[:, 0], br[:, 3] - br[:, 1]), window_ok)
                    by_state[state] += 1
                    cname = tr.class_name
                    by_class[cname] = by_class.get(cname, 0) + 1
                    state_log.setdefault(tr.uid, []).append(
                        (t, state, None if act.estimate is None else act.estimate.speed_bl_s))
                    trail = None
                    if cfg.trail_s > 0:
                        tp = np.array([[(o.box_ref[0] + o.box_ref[2]) / 2, (o.box_ref[1] + o.box_ref[3]) / 2]
                                       for o in tr.obs if o.t > t - cfg.trail_s])
                        trail = apply_h(H_inv, tp) if len(tp) >= 2 else None
                    items.append(VehicleDraw(tr.id, cname, boxes[j], state, trail))
                t0 = lap("activity", t0)

                if len(matched) > peak[0]:
                    peak = (len(matched), round(t - t_first, 3))
                frames_out.append({"t": round(t - t_first, 4), "visible": len(matched), "raw": int(len(scores)),
                                   "camera_ok": step.ok, "inliers": step.n_inliers,
                                   **{s.lower(): by_state[s] for s in STATES}})
                proc_times.append(t)
                panel = PanelInfo(0.0, planned, idx, len(matched), int(len(scores)), tracker.n_confirmed,
                                  step.ok, by_state, by_class)

            # Render every source frame with the latest annotations (one-frame delay for timing).
            if pending is not None:
                writer.write(pending, t - t_first)
            panel.t, panel.frame = t - t_first, idx
            pending = draw_frame(frame, items, panel, (writer.w, writer.h))
            last_t = t
            t0 = lap("render_encode", t0)
            if progress is not None and t - t_first >= next_progress:
                progress(t - t_first, planned)
                next_progress += 2.0
            t0 = time.perf_counter()

        if t_first is None:
            raise ValueError("no frames in the requested time range")
        writer.write(pending, last_t - t_first + 1.0 / info.fps_nominal)
    finally:
        if writer is not None:
            t0 = time.perf_counter()
            writer.close()
            lap("render_encode", t0)

    tracks = tracker.finalize()
    report = _build_report(info, cfg, tracks, state_log, frames_out, np.array(proc_times) - t_first,
                           t_first, last_t, n_decoded, peak, camera_failures)
    report["models"] = {
        "detector": detector.describe() if hasattr(detector, "describe") else {},
        "tracker": {"algorithm": "ByteTrack-style two-stage association in the camera-compensated ground "
                                 "plane, constant-velocity Kalman filter, distance re-find", **asdict(cfg.tracker)},
        "motion_compensation": {"method": "Shi-Tomasi corners (vehicles masked) + pyramidal LK flow + RANSAC, "
                                          "chained to the first processed frame", "model": cfg.motion_model},
        "activity": {"method": "windowed least-squares ground velocity in body lengths/s, 2-sigma margin, "
                               "hysteresis with minimum dwell", **asdict(cfg.activity)},
    }
    if cfg.include_frames:
        report["frames"] = frames_out
    report["output"] = {"video": VIDEO_NAME, "width": writer.w, "height": writer.h, "fps": info.fps_nominal,
                        "frames_written": writer.n_written, "codec": writer.codec,
                        "browser_playable": writer.browser_playable}
    timing["total"] = time.perf_counter() - t_wall0
    report["timing_s"] = {k: round(v, 3) for k, v in timing.items()}
    n_proc = max(1, len(proc_times))
    report["performance"] = {
        "detection_ms_per_processed_frame": round(1e3 * timing.get("detection", 0.0) / n_proc, 2),
        "processed_frames_per_s": round(len(proc_times) / max(1e-9, timing["total"]), 3),
        "realtime_factor": round((last_t - t_first) / max(1e-9, timing["total"]), 4),
    }
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, separators=(",", ":"))
    return report


def _durations(log: List[tuple], proc_t: np.ndarray) -> Dict[str, float]:
    """Seconds per state: each observation is credited until the next processed frame."""
    nxt = {round(float(a), 6): float(b - a) for a, b in zip(proc_t[:-1], proc_t[1:])}
    default = float(np.median(np.diff(proc_t))) if len(proc_t) > 1 else 0.0
    d = {s: 0.0 for s in STATES}
    for t, s, _ in log:
        d[s] += nxt.get(round(float(t), 6), default)
    return d


def _track_record(tr: Track, log: List[tuple], proc_t: np.ndarray, t_first: float, traj_hz: float) -> dict:
    ts = np.array([o.t for o in tr.obs]) - t_first
    br = np.array([o.box_ref for o in tr.obs])
    xy = _centres(br)
    bl = float(np.median(np.maximum(br[:, 2] - br[:, 0], br[:, 3] - br[:, 1])))
    bins = np.floor((ts - ts[0]) / 0.5).astype(int)  # 0.5 s median bins: raw sums mostly measure jitter
    sm = np.array([np.median(xy[bins == b], axis=0) for b in np.unique(bins)])
    path = float(np.linalg.norm(np.diff(sm, axis=0), axis=1).sum()) if len(sm) > 1 else 0.0
    net = float(np.linalg.norm(sm[-1] - sm[0])) if len(sm) > 1 else 0.0
    keep, last = [], -1e9
    for i, t in enumerate(ts):
        if t - last >= 1.0 / traj_hz - 1e-6 or i == len(ts) - 1:
            keep.append(i)
            last = t
    log_rel = [(t - t_first, s, v) for t, s, v in log]
    dur = _durations(log_rel, proc_t)
    speeds = [v for _, _, v in log if v is not None]
    states = [s for _, s, _ in log]
    known = [s for s in states if s != UNKNOWN]
    expected = int(((proc_t >= ts[0] - 1e-9) & (proc_t <= ts[-1] + 1e-9)).sum())
    stages = [o.stage for o in tr.obs]
    return {
        "id": tr.id,
        "class": tr.class_name,
        "class_votes": dict(tr.class_votes),
        "first_t": round(float(ts[0]), 3),
        "last_t": round(float(ts[-1]), 3),
        "duration_s": round(float(ts[-1] - ts[0]), 3),
        "observed_frames": len(tr.obs),
        "observed_fraction": round(len(tr.obs) / max(1, expected), 3),
        "durations_s": {k.lower(): round(v, 3) for k, v in dur.items()},
        "dominant_state": max(set(known), key=known.count) if known else UNKNOWN,
        "state_switches": sum(1 for a, b in zip(states, states[1:]) if a != b and UNKNOWN not in (a, b)),
        "body_length_ref_px": round(bl, 2),
        "path_length_bl": round(path / bl, 2) if bl else None,
        "net_displacement_bl": round(net / bl, 2) if bl else None,
        "median_speed_bl_s": round(float(np.median(speeds)), 4) if speeds else None,
        "max_speed_bl_s": round(float(np.max(speeds)), 4) if speeds else None,
        "trajectory_ref_px": [[round(float(ts[i]), 3), round(float(xy[i, 0]), 1), round(float(xy[i, 1]), 1)]
                              for i in keep],
        "uncertainty": {
            "interruptions": len(tr.gaps),
            "interrupted_s": round(sum(b - a for a, b in tr.gaps), 3),
            "low_confidence_matches": stages.count("low"),
            "refinds": stages.count("refind"),
            "mean_conf": round(float(np.mean([o.conf for o in tr.obs])), 3),
            "end_reason": tr.end_reason,
        },
    }


def _build_report(info, cfg, tracks, state_log, frames_out, proc_t, t_first, last_t, n_decoded, peak,
                  camera_failures) -> dict:
    recs = [_track_record(tr, state_log.get(tr.uid, []), proc_t, t_first, cfg.trajectory_hz) for tr in tracks]
    durs = np.array([r["duration_s"] for r in recs]) if recs else np.zeros(0)
    by_class = {c: 0 for c in VEHICLE_CLASSES}
    for r in recs:
        by_class[r["class"]] = by_class.get(r["class"], 0) + 1
    dom = {s.lower(): sum(1 for r in recs if r["dominant_state"] == s) for s in (MOVING, STATIONARY, UNKNOWN)}
    timeline = []
    for sec in range(int(np.ceil(last_t - t_first)) + 1):
        fs = [f for f in frames_out if sec <= f["t"] < sec + 1]
        if fs:
            m = lambda k: round(float(np.mean([f[k] for f in fs])), 2)  # noqa: E731
            timeline.append({"second": sec, "visible": m("visible"), "moving": m("moving"),
                             "stationary": m("stationary"), "unknown": m("unknown")})
    vis = np.array([f["visible"] for f in frames_out]) if frames_out else np.zeros(0)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": {**info.to_dict(), "processed_range_s": [round(t_first, 3), round(last_t, 3)],
                  "frames_decoded": n_decoded, "frames_processed": len(frames_out),
                  "target_fps": cfg.target_fps},
        "vehicles": {
            "unique_track_ids": len(recs),
            "unique_track_ids_lasting_ge_1s": int((durs >= 1.0).sum()),
            "peak_simultaneous_tracked": int(peak[0]),
            "peak_at_t": peak[1],
            "median_simultaneous_tracked": float(np.median(vis)) if len(vis) else 0.0,
            "by_class": by_class,
            "tracks_by_dominant_state": dom,
            "camera_registration_failures": camera_failures,
            "note": ("Counts are model estimates, not validated against ground truth. IDs over-count when "
                     "tracks fragment or vehicles re-enter; speeds are in body lengths per second (no metric "
                     "scale); state confidence is not a calibrated probability."),
        },
        "timeline_per_second": timeline,
        "tracks": recs,
    }
