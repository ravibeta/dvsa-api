"""Query-time aggregation over raw commentary events.

The observability layer stores events **raw and wide** and defers all roll-ups to
read time (framework principle: *query-time aggregation*). This module is the
pure, Django-free engine that powers the aggregate API endpoint — it operates on
plain event dicts (whatever the store returns) so it is trivially unit-testable
and equally usable over an in-memory list, a DB queryset's ``.values()``, or an
OTLP export.

Example
-------
>>> aggregate_events(events, group_by="source",
...                   metrics=["count", "sum:count", "avg:mean_score"])
{'groups': {...}, 'total_events': 42}
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# Supported reducers over a list of numeric values.
_REDUCERS = {
    "sum": lambda xs: float(sum(xs)),
    "avg": lambda xs: float(sum(xs) / len(xs)) if xs else 0.0,
    "min": lambda xs: float(min(xs)) if xs else 0.0,
    "max": lambda xs: float(max(xs)) if xs else 0.0,
    "count": lambda xs: float(len(xs)),
}


def _as_dict(event: Any) -> Dict[str, Any]:
    """Accept either a CommentaryEvent or a plain dict."""

    if hasattr(event, "to_dict"):
        return event.to_dict()
    return dict(event)


def _get_field(event: Dict[str, Any], path: str) -> Any:
    """Resolve ``path`` against an event.

    Supports top-level fields (``source``, ``frame_index``) and one level of
    nesting into the JSON maps via dotted paths (``attributes.label``,
    ``metrics.count``, ``metadata.routine``).
    """

    if "." in path:
        head, tail = path.split(".", 1)
        container = event.get(head)
        if isinstance(container, dict):
            return container.get(tail)
        return None
    return event.get(path)


def _matches(event: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if _get_field(event, key) != expected:
            return False
    return True


def _parse_metric_spec(spec: str):
    """``"avg:mean_score"`` -> ("avg", "mean_score"); ``"count"`` -> ("count", None)."""

    if spec == "count":
        return "count", None
    if ":" not in spec:
        # Bare metric name defaults to sum.
        return "sum", spec
    op, _, field = spec.partition(":")
    if op not in _REDUCERS:
        raise ValueError("Unknown aggregation op '%s' in '%s'" % (op, spec))
    return op, field


def aggregate_events(
    events: Iterable[Any],
    *,
    group_by: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate ``events`` at query time.

    Parameters
    ----------
    events:
        Iterable of ``CommentaryEvent`` or event dicts.
    group_by:
        Field path to group on (e.g. ``"source"``, ``"frame_index"``,
        ``"metadata.routine"``). ``None`` groups everything into ``"all"``.
    metrics:
        List of metric specs. Each is ``"count"``, a bare metric name (summed),
        or ``"<op>:<metric>"`` where op ∈ {sum, avg, min, max, count} and
        ``<metric>`` is a key in the event's ``metrics`` map. Defaults to
        ``["count"]``.
    filters:
        Optional exact-match field filters applied before grouping.

    Returns
    -------
    ``{"groups": {group_value: {metric_label: number, "events": n}}, "total_events": N}``
    """

    metrics = metrics or ["count"]
    specs = [(spec, _parse_metric_spec(spec)) for spec in metrics]

    # Bucket the raw metric values per group so reducers run once at the end.
    buckets: Dict[Any, Dict[str, List[float]]] = {}
    counts: Dict[Any, int] = {}
    total = 0

    for raw in events:
        event = _as_dict(raw)
        if not _matches(event, filters):
            continue
        total += 1
        gval = "all" if group_by is None else _get_field(event, group_by)
        # Normalise unhashable / None group keys to a stable string.
        if gval is None:
            gval = "null"
        counts[gval] = counts.get(gval, 0) + 1
        group_metrics = event.get("metrics", {}) or {}
        bucket = buckets.setdefault(gval, {})
        for label, (op, mfield) in specs:
            if op == "count":
                bucket.setdefault(label, []).append(1.0)
            else:
                val = group_metrics.get(mfield)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    bucket.setdefault(label, []).append(float(val))

    groups: Dict[Any, Dict[str, Any]] = {}
    for gval, bucket in buckets.items():
        out: Dict[str, Any] = {"events": counts[gval]}
        for label, (op, _mfield) in specs:
            out[label] = _REDUCERS[op](bucket.get(label, []))
        groups[gval] = out

    return {"groups": groups, "total_events": total}
