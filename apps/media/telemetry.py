"""Capture drone position from uploaded telemetry (DJI-style ``.srt`` sidecars or metadata).

Only *captures* what was uploaded so job results can place detections near the drone; it
does not georeference frames. The latitude/longitude patterns are the ones the ``/corners/``
runner already uses (:mod:`core.azure.survey_corners`).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from core.azure.survey_corners import SRT_LAT, SRT_LON

SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")
SRT_ALT = re.compile(r"\[(?:rel_alt|altitude)\s*:\s*([-\d.]+)")
MAX_SAMPLES = 5000


def _valid(lat: float, lon: float) -> bool:
    return (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90
            and -180 <= lon <= 180 and not (lat == 0 and lon == 0))


def parse_srt(text: str) -> List[Dict[str, float]]:
    """``[{t, lat, lon[, alt]}]`` (``t`` = cue start in seconds), thinned to ``MAX_SAMPLES``."""
    samples: List[Dict[str, float]] = []
    for block in re.split(r"\r?\n\s*\r?\n", text or ""):
        when, lat, lon = SRT_TIME.search(block), SRT_LAT.search(block), SRT_LON.search(block)
        if not (when and lat and lon):
            continue
        try:
            h, m, s, ms = when.groups()
            sample = {"t": round(int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000, 3),
                      "lat": float(lat.group(1)), "lon": float(lon.group(1))}
            alt = SRT_ALT.search(block)
            if alt:
                sample["alt"] = float(alt.group(1))
        except ValueError:
            continue
        if _valid(sample["lat"], sample["lon"]):
            samples.append(sample)
    if len(samples) > MAX_SAMPLES:
        step = math.ceil(len(samples) / MAX_SAMPLES)
        samples = samples[::step]
    return samples


def from_metadata(metadata: Optional[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Samples from client metadata: ``telemetry: [{t, lat, lon}]`` or a single
    ``latitude``/``longitude`` (taken as the position for the whole video)."""
    metadata = metadata or {}
    out: List[Dict[str, float]] = []
    for item in metadata.get("telemetry") or []:
        try:
            sample = {"t": float(item.get("t", 0)), "lat": float(item["lat"]), "lon": float(item["lon"])}
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if _valid(sample["lat"], sample["lon"]):
            out.append(sample)
    if not out:
        try:
            lat, lon = float(metadata["latitude"]), float(metadata["longitude"])
        except (KeyError, TypeError, ValueError):
            return []
        if _valid(lat, lon):
            out.append({"t": 0.0, "lat": lat, "lon": lon})
    return out


def nearest(samples: List[Dict[str, float]], t: float) -> Optional[Dict[str, float]]:
    return min(samples, key=lambda s: abs(s["t"] - t)) if samples else None
