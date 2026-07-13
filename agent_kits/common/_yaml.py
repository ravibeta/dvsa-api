"""Tiny, dependency-optional YAML loader for the kits' config files.

Uses PyYAML's ``safe_load`` when installed. Otherwise falls back to a minimal,
indentation-based parser that handles exactly the subset used by the kit's
``agent_profile.yaml`` and ``*.skill`` files: nested mappings, scalar lists,
comments, quoted scalars, and folded (``>``) block scalars. This keeps the kits
importable and testable without a third-party dependency.
"""

from __future__ import annotations

from typing import Any, List, Tuple


def safe_load(text: str) -> Any:
    """Parse ``text`` as YAML, preferring PyYAML, else the built-in fallback."""
    try:
        import yaml  # noqa: PLC0415 - optional dependency
        return yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - not installed OR a parse error we retry below
        return _mini_load(text)


def load_file(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return safe_load(fh.read())


# --------------------------------------------------------------------------- #
# Minimal fallback parser (controlled subset only).
# --------------------------------------------------------------------------- #
def _mini_load(text: str) -> Any:
    lines = _logical_lines(text)
    value, _ = _parse_block(lines, 0, indent=0)
    return value


def _logical_lines(text: str) -> List[Tuple[int, str]]:
    """Return (indent, content) for each significant line (comments stripped)."""
    out: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].rstrip() if not _in_quote(raw) else raw.rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        out.append((indent, stripped.strip()))
    return out


def _in_quote(line: str) -> bool:
    s = line.strip()
    return s.startswith(('"', "'"))


def _parse_block(lines: List[Tuple[int, str]], i: int, indent: int) -> Tuple[Any, int]:
    if i >= len(lines):
        return None, i
    # List block?
    if lines[i][1].startswith("- "):
        items: List[Any] = []
        while i < len(lines) and lines[i][0] >= indent and lines[i][1].startswith("- "):
            if lines[i][0] != indent:
                break
            items.append(_scalar(lines[i][1][2:].strip()))
            i += 1
        return items, i
    # Mapping block.
    mapping: dict = {}
    while i < len(lines) and lines[i][0] == indent and not lines[i][1].startswith("- "):
        key, _, rest = lines[i][1].partition(":")
        key = key.strip()
        rest = rest.strip()
        i += 1
        if rest in (">", "|", ">-", "|-"):
            folded, i = _folded(lines, i, indent)
            mapping[key] = folded
        elif rest == "":
            child_indent = lines[i][0] if i < len(lines) else indent
            if i < len(lines) and child_indent > indent:
                value, i = _parse_block(lines, i, child_indent)
                mapping[key] = value
            else:
                mapping[key] = None
        else:
            mapping[key] = _scalar(rest)
    return mapping, i


def _folded(lines: List[Tuple[int, str]], i: int, parent_indent: int) -> Tuple[str, int]:
    parts: List[str] = []
    while i < len(lines) and lines[i][0] > parent_indent:
        parts.append(lines[i][1])
        i += 1
    return " ".join(parts), i


def _scalar(token: str) -> Any:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    low = token.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~", ""):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


__all__ = ["safe_load", "load_file"]
