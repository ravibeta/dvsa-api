"""Deterministic urban-street accident detector (pure Python, no ML deps).

Demonstrates the full bring-your-own-reasoning pipeline on drone video *metadata*
(object tracks), with no heavy dependencies. The heuristic flags an ``accident``
when both hold:

1. **Convergence** — two or more tracks come within ``proximity_threshold``
   pixels of each other inside a short ``time_window_s`` window; and
2. **Sudden deceleration** — at least one of those tracks shows a speed drop
   between consecutive samples exceeding ``deceleration_threshold``.

Each track is a dict::

    {
      "id": "veh-1",
      "class": "vehicle",
      "samples": [{"t": 0.0, "x": 10.0, "y": 5.0}, ...]   # or with "vx"/"vy"
    }

Velocities are used if present, otherwise derived from consecutive positions.
The output is contract-compliant ``{actions, reasoning_trace, metadata}``.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

MODEL_NAME = "urban_accident_example"
MODEL_VERSION = "1.0.0"


class ReasoningModelAdapter:
    """Rule-based accident detector implementing the reasoning contract."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.proximity_threshold = float(config.get("proximity_threshold", 8.0))
        self.time_window_s = float(config.get("time_window_s", 1.0))
        self.deceleration_threshold = float(config.get("deceleration_threshold", 6.0))

    # ----- contract -----------------------------------------------------
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        tracks = context.get("tracks") or []
        trace: List[str] = [
            f"query: {context.get('query', 'detect urban-street anomalies')!r}",
            f"analysing {len(tracks)} object track(s)",
        ]

        converged = self._find_convergence(tracks, trace)
        decel_ids = self._find_decelerations(tracks, trace)

        actions: List[Dict[str, Any]] = []
        if converged and (decel_ids & {converged[0], converged[1]}):
            point = converged[2]
            bbox = [round(point[0] - 5, 3), round(point[1] - 5, 3), 10.0, 10.0]
            trace.append(
                f"tracks {converged[0]} and {converged[1]} converge at "
                f"({point[0]:.2f}, {point[1]:.2f}) with sudden deceleration "
                f"-> accident")
            actions.append({
                "type": "anomaly",
                "label": "accident",
                "confidence": 0.95,
                "bbox": bbox,
                "tracks": [converged[0], converged[1]],
            })
        else:
            trace.append("no convergence + deceleration coincidence -> nominal scene")
            actions.append({"type": "label", "label": "nominal", "confidence": 0.6})

        return {
            "actions": actions,
            "reasoning_trace": trace,
            "metadata": {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "tokens": None,
                "latency_ms": int((time.time() - started) * 1000),
                "request_id": uuid.uuid4().hex,
                "reasoning_effort": context.get("reasoning_effort", "low"),
            },
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION}

    # ----- heuristics ---------------------------------------------------
    def _find_convergence(
        self, tracks: List[Dict[str, Any]], trace: List[str],
    ) -> Optional[Tuple[str, str, Tuple[float, float]]]:
        """Return ``(id_a, id_b, midpoint)`` of the closest converging pair."""
        best: Optional[Tuple[float, str, str, Tuple[float, float]]] = None
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                pair = _closest_approach(
                    _samples(a), _samples(b), self.time_window_s)
                if pair is None:
                    continue
                dist, point = pair
                if dist <= self.proximity_threshold:
                    if best is None or dist < best[0]:
                        best = (dist, _track_id(a), _track_id(b), point)
        if best is None:
            return None
        trace.append(
            f"closest approach {best[1]}<->{best[2]} = {best[0]:.2f}px "
            f"(<= {self.proximity_threshold}px)")
        return best[1], best[2], best[3]

    def _find_decelerations(
        self, tracks: List[Dict[str, Any]], trace: List[str],
    ) -> set:
        """Return the set of track ids exhibiting a sudden speed drop."""
        flagged = set()
        for track in tracks:
            drop = _max_speed_drop(_samples(track))
            if drop >= self.deceleration_threshold:
                flagged.add(_track_id(track))
                trace.append(
                    f"track {_track_id(track)} decelerates by {drop:.2f} "
                    f"(>= {self.deceleration_threshold})")
        return flagged


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def _track_id(track: Dict[str, Any]) -> str:
    return str(track.get("id", "unknown"))


def _samples(track: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(track.get("samples") or track.get("positions") or [])


def _speed_at(samples: List[Dict[str, Any]], idx: int) -> float:
    """Speed at sample ``idx`` — use provided vx/vy, else finite differences."""
    s = samples[idx]
    if "vx" in s or "vy" in s:
        return math.hypot(float(s.get("vx", 0.0)), float(s.get("vy", 0.0)))
    if idx == 0:
        return 0.0
    prev = samples[idx - 1]
    dt = float(s.get("t", idx)) - float(prev.get("t", idx - 1))
    if dt <= 0:
        return 0.0
    return math.hypot(float(s["x"]) - float(prev["x"]),
                      float(s["y"]) - float(prev["y"])) / dt


def _max_speed_drop(samples: List[Dict[str, Any]]) -> float:
    """Largest speed decrease between consecutive samples (0 if <2 samples)."""
    if len(samples) < 2:
        return 0.0
    speeds = [_speed_at(samples, i) for i in range(len(samples))]
    return max((speeds[i] - speeds[i + 1] for i in range(len(speeds) - 1)),
               default=0.0)


def _closest_approach(
    a: List[Dict[str, Any]], b: List[Dict[str, Any]], window_s: float,
) -> Optional[Tuple[float, Tuple[float, float]]]:
    """Min distance between two tracks at near-equal timestamps within a window."""
    best: Optional[Tuple[float, Tuple[float, float]]] = None
    for sa in a:
        for sb in b:
            ta, tb = float(sa.get("t", 0.0)), float(sb.get("t", 0.0))
            if abs(ta - tb) > window_s:
                continue
            dist = math.hypot(float(sa["x"]) - float(sb["x"]),
                              float(sa["y"]) - float(sb["y"]))
            midpoint = ((float(sa["x"]) + float(sb["x"])) / 2.0,
                        (float(sa["y"]) + float(sb["y"])) / 2.0)
            if best is None or dist < best[0]:
                best = (dist, midpoint)
    return best
