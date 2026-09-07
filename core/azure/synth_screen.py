"""Flag obviously fabricated aerial drone footage at upload time.

Ported from the workspace reference script ``drone_synth_screen.py`` into a
reusable, side-effect-free runner so it can gate the video-upload path directly:
a clip that screens as fabricated is refused before it is ever stored or indexed.

The screening is tuned for precision, unchanged from the script. The default
verdict is ``camera``; a clip is only flagged when several independent signs of
fabrication agree AND no positive sign of physical capture is present:

  synthetic-declared   generator tag or C2PA AI manifest found in the container
  likely-fabricated    >= MIN_SYNTH_SIGNS fabrication signs, no camera evidence
  review               fewer signs, still no camera evidence
  camera               positive evidence of physical capture (the default)

``FABRICATED_VERDICTS`` is the set the upload path rejects on. ``ffprobe`` and
``cv2`` are invoked lazily inside the functions that need them, so importing this
module stays cheap and the pure verdict logic (:func:`decide`) is unit-testable
without a video runtime.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("apps.videos")

# Verdicts that mean "not camera-based" — the upload path refuses these.
FABRICATED_VERDICTS = frozenset({"likely-fabricated", "synthetic-declared"})

# --------------------------------------------------------------- calibration

# Positive evidence of physical capture. Any one of these vetoes a flag, so
# they are deliberately set where real footage clears them comfortably.
CAMERA_EVIDENCE = {
    "sensor_pattern": 0.020,      # residual_corr >= this: a fixed sensor pattern
}

# Signs of fabrication. Set well past where real footage sits, so that a clip
# has to be clearly anomalous on each one.
SYNTH_SIGNS = {
    "sensor_pattern": 0.006,      # residual_corr < this: no sensor pattern at all
    "smooth_residual": 0.90,      # residual_energy < this: too little sensor grain
    "incoherent_inliers": 0.85,   # inlier_ratio < this
    "warp_reproj": 1.35,          # reproj_error > this
}

MIN_SYNTH_SIGNS = 3               # how many must agree before flagging

# track_survival and motion_jerk are computed and reported but deliberately NOT
# used in the verdict. On controlled probes both were unreliable or inverted:
# fast-yaw real footage had the LOWEST track survival of any clip, and
# fabricated clips had SMOOTHER motion than every real clip. Wire them in only
# if the script's --calibrate shows real separation on your own corpus.

GENERATOR_HINTS = re.compile(
    r"sora|runway|gen-?[23]|pika|luma|dream ?machine|kling|veo|stable[- ]?video|"
    r"svd|animatediff|midjourney|firefly|synthes(is|ia)|trainedAlgorithmicMedia",
    re.I,
)
CAMERA_HINTS = re.compile(r"dji|autel|parrot|skydio|gopro|yuneec|hasselblad|fc\d{3,4}", re.I)


# --------------------------------------------------------------- provenance

def ffprobe(path: str) -> Dict[str, Any]:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        return json.loads(out) if out.strip() else {}
    except Exception:
        return {}


def scan_raw_boxes(path: str, limit: int = 4_000_000) -> Dict[str, bool]:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(limit)
            f.seek(max(0, size - limit))
            tail = f.read(limit)
    except OSError:
        return {"c2pa_present": False, "c2pa_ai_manifest": False}
    blob = head + tail
    has_c2pa = b"c2pa" in blob or b"jumb" in blob.lower()
    ai_claim = False
    if has_c2pa or b"<x:xmpmeta" in blob:
        text = blob.decode("latin-1", errors="ignore")
        ai_claim = bool(GENERATOR_HINTS.search(text))
    return {"c2pa_present": has_c2pa, "c2pa_ai_manifest": ai_claim}


def provenance_features(path: str) -> Dict[str, Any]:
    probe = ffprobe(path)
    fmt = probe.get("format", {}) or {}
    streams = probe.get("streams", []) or []
    tags = {k.lower(): str(v) for k, v in (fmt.get("tags") or {}).items()}
    for s in streams:
        for k, v in (s.get("tags") or {}).items():
            tags.setdefault(f"{s.get('codec_type', '?')}.{k.lower()}", str(v))
    blob = " ".join(tags.values())
    vstream = next((s for s in streams if s.get("codec_type") == "video"), {})

    feats = {
        "duration_s": round(float(fmt.get("duration") or 0.0), 3),
        "width": vstream.get("width"),
        "height": vstream.get("height"),
        "codec": vstream.get("codec_name"),
        "encoder_tag": tags.get("encoder", ""),
        # Reported for context only -- never scored in either direction.
        "has_camera_metadata": bool(CAMERA_HINTS.search(blob)),
        "has_telemetry": (
            any("location" in k or "gps" in k for k in tags)
            or any(s.get("codec_type") in ("data", "subtitle") for s in streams)
            or os.path.exists(os.path.splitext(path)[0] + ".srt")
        ),
        "generator_tag": bool(GENERATOR_HINTS.search(blob)),
    }
    feats.update(scan_raw_boxes(path))
    return feats


# ---------------------------------------------------------- frame sampling

def sample_bursts(path: str, n_bursts: int = 4, burst_len: int = 24, width: int = 640):
    """Return short runs of CONSECUTIVE frames spread across the video.

    Consecutive frames are essential: geometry, tracking and motion features
    are meaningless across a multi-second gap.
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total < burst_len:
        n_bursts, burst_len = 1, max(total, 0)
    starts = np.linspace(0, max(total - burst_len - 1, 0), n_bursts).astype(int)

    bursts = []
    for st in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(st))
        run = []
        for _ in range(burst_len):
            ok, f = cap.read()
            if not ok:
                break
            scale = width / float(f.shape[1])
            if scale < 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            run.append(f)
        if len(run) >= 3:
            bursts.append(run)
    cap.release()
    return bursts


# ------------------------------------------------------------ signal probes

def sensor_features(bursts) -> Dict[str, Any]:
    """A physical sensor leaves a pattern shared across frames; a generator does not."""
    import cv2  # noqa: PLC0415

    frames = [b[i] for b in bursts for i in (0, len(b) // 2, len(b) - 1)]
    if len(frames) < 3:
        return {"residual_energy": None, "residual_corr": None}
    res = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        res.append(g - cv2.bilateralFilter(g, 5, 30, 30))
    energy = float(np.mean([np.mean(np.abs(r)) for r in res]))

    shape = res[0].shape
    res = [r for r in res if r.shape == shape]
    stack = np.stack([r - r.mean() for r in res])
    n = len(stack)
    corrs = []
    for i in range(n):                       # leave-one-out, avoids self-correlation
        other = (stack.sum(axis=0) - stack[i]) / (n - 1)
        a, b = stack[i].ravel(), other.ravel()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        if d > 1e-6:
            corrs.append(float(np.dot(a, b) / d))
    return {
        "residual_energy": round(energy, 4),
        "residual_corr": round(float(np.mean(corrs)), 5) if corrs else None,
    }


def geometry_features(bursts) -> Dict[str, Any]:
    """Consecutive-frame homography fit. Real scenes are rigid and align well."""
    import cv2  # noqa: PLC0415

    orb = cv2.ORB_create(1200)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ratios, errors = [], []
    for burst in bursts:
        for a, b in zip(burst[:-1], burst[1:]):
            ka, da = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
            kb, db = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
            if da is None or db is None or len(da) < 12 or len(db) < 12:
                continue
            m = matcher.match(da, db)
            if len(m) < 12:
                continue
            src = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
            dst = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is None or mask is None:
                continue
            mask = mask.ravel().astype(bool)
            ratios.append(float(mask.mean()))
            if mask.sum() >= 4:
                proj = cv2.perspectiveTransform(src[mask], H)
                errors.append(float(np.mean(np.linalg.norm(proj - dst[mask], axis=2))))
    return {
        "inlier_ratio": round(float(np.median(ratios)), 4) if ratios else None,
        "reproj_error": round(float(np.median(errors)), 4) if errors else None,
    }


def tracking_features(bursts) -> Dict[str, Any]:
    """Ground features persist in real footage; generated content morphs under them."""
    import cv2  # noqa: PLC0415

    survivals = []
    for burst in bursts:
        g0 = cv2.cvtColor(burst[0], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(g0, maxCorners=300, qualityLevel=0.01,
                                     minDistance=8)
        if p0 is None or len(p0) < 20:
            continue
        n0, prev, pts = len(p0), g0, p0
        alive = np.ones(n0, dtype=bool)
        for f in burst[1:]:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None,
                                                  winSize=(21, 21), maxLevel=3)
            if nxt is None:
                break
            st = st.ravel().astype(bool)
            # forward-backward check rejects tracks that drift onto new content
            back, _, _ = cv2.calcOpticalFlowPyrLK(g, prev, nxt, None,
                                                  winSize=(21, 21), maxLevel=3)
            if back is not None:
                fb = np.linalg.norm(back - pts, axis=2).ravel()
                st &= fb < 1.5
            idx = np.where(alive)[0]
            alive[idx[~st]] = False
            if not alive.any():
                break
            pts, prev = nxt[st].reshape(-1, 1, 2), g
        survivals.append(float(alive.sum()) / n0)
    return {"track_survival": round(float(np.mean(survivals)), 4) if survivals else None}


def motion_features(bursts) -> Dict[str, Any]:
    """Flight dynamics are smooth; fabricated camera motion tends to be erratic."""
    import cv2  # noqa: PLC0415

    ratios = []
    for burst in bursts:
        steps = []
        for a, b in zip(burst[:-1], burst[1:]):
            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(ga, maxCorners=250, qualityLevel=0.01,
                                         minDistance=8)
            if p0 is None or len(p0) < 15:
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None,
                                                 winSize=(21, 21), maxLevel=3)
            if p1 is None:
                continue
            st = st.ravel().astype(bool)
            if st.sum() < 10:
                continue
            steps.append(np.median((p1[st] - p0[st]).reshape(-1, 2), axis=0))
        if len(steps) < 5:
            continue
        s = np.array(steps)
        vel = np.linalg.norm(s, axis=1)
        accel = np.linalg.norm(np.diff(s, axis=0), axis=1)
        denom = float(np.mean(vel)) + 0.25          # guard near-hover clips
        ratios.append(float(np.mean(accel) / denom))
    return {"motion_jerk": round(float(np.median(ratios)), 4) if ratios else None}


# ------------------------------------------------------------------ verdict

def decide(f: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    cam, syn = [], []
    rc, ir = f.get("residual_corr"), f.get("inlier_ratio")
    rp, re_ = f.get("reproj_error"), f.get("residual_energy")

    # --- veto-grade evidence of physical capture ---
    # Only signatures a generator cannot produce count here. Scene coherence
    # does NOT: a good generator makes coherent video, so rigid geometry and
    # persistent features prove nothing about origin.
    if rc is not None and rc >= CAMERA_EVIDENCE["sensor_pattern"]:
        cam.append("sensor_pattern_noise")
    if f.get("has_camera_metadata") or f.get("has_telemetry"):
        cam.append("camera_provenance")

    # --- signs of fabrication ---
    if rc is not None and rc < SYNTH_SIGNS["sensor_pattern"]:
        syn.append("no_sensor_pattern")
    if re_ is not None and re_ < SYNTH_SIGNS["smooth_residual"]:
        syn.append("too_little_grain")
    if ir is not None and ir < SYNTH_SIGNS["incoherent_inliers"]:
        syn.append("incoherent_scene")
    if rp is not None and rp > SYNTH_SIGNS["warp_reproj"]:
        syn.append("non_rigid_warp")

    if f.get("generator_tag") or f.get("c2pa_ai_manifest"):
        verdict = "synthetic-declared"
    elif cam:
        verdict = "camera"
    elif len(syn) >= MIN_SYNTH_SIGNS:
        verdict = "likely-fabricated"
    elif syn:
        verdict = "review"
    else:
        verdict = "camera"
    return verdict, cam, syn


def screen(path: str, n_bursts: int = 4, burst_len: int = 24,
           width: int = 640) -> Dict[str, Any]:
    """Screen the video at ``path`` and return its feature record + verdict.

    The record's ``verdict`` is one of ``synthetic-declared`` / ``likely-fabricated``
    / ``review`` / ``camera``; ``fabrication_signs`` and ``camera_evidence`` list
    the reasons behind it. Missing/absent evidence is never scored, so a clip that
    cannot be decoded lands on the default ``camera`` verdict.
    """
    f: Dict[str, Any] = {"path": path}
    f.update(provenance_features(path))
    bursts = sample_bursts(path, n_bursts, burst_len, width)
    f["bursts"] = len(bursts)
    if bursts:
        f.update(sensor_features(bursts))
        f.update(geometry_features(bursts))
        f.update(tracking_features(bursts))
        f.update(motion_features(bursts))
    verdict, cam, syn = decide(f)
    f["verdict"] = verdict
    f["camera_evidence"] = cam
    f["fabrication_signs"] = syn
    return f
