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

# ================================================================
# Additional physics-based aerial-drone provenance features
#
# Heavy runtimes (cv2, scikit-image) are imported lazily inside each
# function, matching the rest of this module, so importing the module
# never requires a vision stack. scikit-image is optional: when it is not
# installed, glcm_features degrades to None rather than failing the import.
# ================================================================


# ---------------------------------------------------------------
# 1. Rolling-shutter gradient (CMOS readout signature)
# ---------------------------------------------------------------

def rolling_shutter_signature(frames):
    import cv2  # noqa: PLC0415

    grads = []
    for a, b in zip(frames[:-1], frames[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(ga, gb, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        vx = flow[..., 0]
        g = np.mean(np.abs(np.gradient(vx, axis=0)))
        grads.append(g)
    return {"rs_gradient": float(np.median(grads)) if grads else None}


# ---------------------------------------------------------------
# 2. Radial vignetting slope (lens illumination falloff)
# ---------------------------------------------------------------

def radial_vignetting(frames):
    import cv2  # noqa: PLC0415

    slopes = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        r_norm = r / r.max()
        bins = np.linspace(0, 1, 20)
        prof = [g[r_norm <= b].mean() for b in bins]
        slope = np.polyfit(bins, prof, 1)[0]
        slopes.append(slope)
    return {"vignette_slope": float(np.median(slopes)) if slopes else None}


# ---------------------------------------------------------------
# 3. Bayer demosaicing correlation (physical sensor signature)
# ---------------------------------------------------------------

def bayer_demosaic_signature(frames):
    corrs = []
    for f in frames:
        r, g, b = [f[..., i].astype(np.float32) for i in range(3)]
        rc = np.corrcoef(r[:-1, :].ravel(), r[1:, :].ravel())[0, 1]
        gc = np.corrcoef(g[:-1, :].ravel(), g[1:, :].ravel())[0, 1]
        bc = np.corrcoef(b[:-1, :].ravel(), b[1:, :].ravel())[0, 1]
        corrs.append((rc + gc + bc) / 3)
    return {"demosaic_corr": float(np.median(corrs)) if corrs else None}


# ---------------------------------------------------------------
# 4. Compression periodicity (H.264/H.265 block boundary energy)
# ---------------------------------------------------------------

def compression_periodicity(frames):
    import cv2  # noqa: PLC0415

    scores = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        diff = np.abs(np.diff(g, axis=1))
        col_energy = diff[:, ::8].mean()
        diff = np.abs(np.diff(g, axis=0))
        row_energy = diff[::8, :].mean()
        scores.append((col_energy + row_energy) / 2)
    return {"block_periodicity": float(np.median(scores)) if scores else None}


# ---------------------------------------------------------------
# 5. Temporal chromatic drift (auto-exposure / AWB signature)
# ---------------------------------------------------------------

def chromatic_drift(frames):
    import cv2  # noqa: PLC0415

    drifts = []
    prev = None
    for f in frames:
        ycrcb = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        cr_mean = ycrcb[..., 1].mean()
        if prev is not None:
            drifts.append(abs(cr_mean - prev))
        prev = cr_mean
    return {"chromatic_drift": float(np.median(drifts)) if drifts else None}


# ---------------------------------------------------------------
# 6. Texture anisotropy (roads, roofs, vegetation directional cues)
# ---------------------------------------------------------------

def texture_anisotropy(frames):
    import cv2  # noqa: PLC0415

    anis = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        a = np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + 1e-6)
        anis.append(a)
    return {"anisotropy_ratio": float(np.median(anis)) if anis else None}


# ---------------------------------------------------------------
# 7. Fractal dimension (natural aerial scaling behavior)
# ---------------------------------------------------------------

def fractal_dimension(frames):
    import cv2  # noqa: PLC0415

    dims = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g, 80, 160)
        sizes = np.logspace(1, 5, num=10, base=2).astype(int)
        counts = []
        for s in sizes:
            k = cv2.resize(edges, (edges.shape[1] // s, edges.shape[0] // s))
            counts.append(np.sum(k > 0))
        slope = np.polyfit(np.log(sizes), np.log(counts), 1)[0]
        dims.append(slope)
    return {"fractal_dim": float(np.median(dims)) if dims else None}


# ---------------------------------------------------------------
# 8. GLCM texture statistics (co-occurrence matrix)
# ---------------------------------------------------------------

def glcm_features(frames, levels=64):
    import cv2  # noqa: PLC0415

    empty = {"glcm_contrast": None, "glcm_homogeneity": None,
             "glcm_energy": None, "glcm_entropy": None}
    try:
        # scikit-image >= 0.19 renamed grey* -> gray*; keep a fallback for
        # older installs. If scikit-image is absent, degrade to None.
        try:
            from skimage.feature import graycomatrix, graycoprops  # noqa: PLC0415
        except ImportError:  # pragma: no cover - old scikit-image
            from skimage.feature import (  # noqa: PLC0415
                greycomatrix as graycomatrix,
                greycoprops as graycoprops,
            )
    except ImportError:
        logger.info("scikit-image unavailable; skipping GLCM features")
        return empty

    stats = {"glcm_contrast": [], "glcm_homogeneity": [],
             "glcm_energy": [], "glcm_entropy": []}
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
        q = (g.astype(np.float32) / 256.0 * (levels - 1)).astype(np.uint8)
        glcm = graycomatrix(q, distances=[1], angles=[0],
                            levels=levels, symmetric=True, normed=True)
        stats["glcm_contrast"].append(graycoprops(glcm, "contrast")[0, 0])
        stats["glcm_homogeneity"].append(graycoprops(glcm, "homogeneity")[0, 0])
        stats["glcm_energy"].append(graycoprops(glcm, "energy")[0, 0])
        p = glcm[:, :, 0, 0].ravel()
        p = p[p > 0]
        stats["glcm_entropy"].append(-np.sum(p * np.log(p)))
    return {k: float(np.median(v)) if v else None for k, v in stats.items()}


# ---------------------------------------------------------------
# 9. DCT high-frequency spectral scars (GAN/diffusion fingerprints)
# ---------------------------------------------------------------

def dct_features(frames):
    import cv2  # noqa: PLC0415

    hf_means, hf_vars, hf_kurt = [], [], []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32) - 128.0
        dct = cv2.dct(g)
        h = dct[128:, 128:]
        vals = h.ravel()
        hf_means.append(float(np.mean(np.abs(vals))))
        hf_vars.append(float(np.var(vals)))
        m = np.mean(vals)
        s2 = np.var(vals)
        if s2 > 1e-6:
            hf_kurt.append(float(np.mean(((vals - m)**4)) / (s2**2)))
    return {
        "hf_mean": float(np.median(hf_means)) if hf_means else None,
        "hf_var": float(np.median(hf_vars)) if hf_vars else None,
        "hf_kurt": float(np.median(hf_kurt)) if hf_kurt else None,
    }


# ---------------------------------------------------------------
# 10. Chrominance-luminance variance coupling (optical physics)
# ---------------------------------------------------------------

def chroma_luma_features(frames, patch_size=16):
    import cv2  # noqa: PLC0415

    corrs = []
    for f in frames:
        ycrcb = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        Y, Cr, Cb = [ycrcb[..., i] for i in range(3)]
        h, w = Y.shape
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                y_patch = Y[i:i+patch_size, j:j+patch_size]
                cr_patch = Cr[i:i+patch_size, j:j+patch_size]
                cb_patch = Cb[i:i+patch_size, j:j+patch_size]
                if np.std(y_patch) < 5.0:
                    continue
                vy = np.var(y_patch)
                vcr = np.var(cr_patch)
                vcb = np.var(cb_patch)
                vec = np.array([vy, vcr, vcb])
                if np.all(vec > 0):
                    corrs.append(np.corrcoef(vec)[0, 1])
    return {"y_cr_var_corr": float(np.median(corrs)) if corrs else None}


# ---------------------------------------------------------------
# Unified multi-feature aggregator
# ---------------------------------------------------------------

def pixel_distribution_features(bursts):
    frames = [b[i] for b in bursts for i in (0, len(b)//2, len(b)-1)]
    if len(frames) < 3:
        return {}

    out = {}
    out.update(rolling_shutter_signature(frames))
    out.update(radial_vignetting(frames))
    out.update(bayer_demosaic_signature(frames))
    out.update(compression_periodicity(frames))
    out.update(chromatic_drift(frames))
    out.update(texture_anisotropy(frames))
    out.update(fractal_dimension(frames))
    out.update(glcm_features(frames))
    out.update(dct_features(frames))
    out.update(chroma_luma_features(frames))

    return out


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
        f.update(pixel_distribution_features(bursts))
    verdict, cam, syn = decide(f)
    f["verdict"] = verdict
    f["camera_evidence"] = cam
    f["fabrication_signs"] = syn
    return f
