"""Locate ffmpeg / ffprobe executables.

Order: explicit argument -> ``DVSA_FFMPEG`` / ``DVSA_FFPROBE`` environment variable -> ``PATH`` ->
(ffmpeg only) the static binary bundled with the ``imageio-ffmpeg`` wheel.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional


def _existing(path: Optional[str]) -> Optional[str]:
    return path if path and os.path.isfile(path) else None


def find_ffmpeg(explicit: Optional[str] = None) -> Optional[str]:
    found = _existing(explicit) or _existing(os.environ.get("DVSA_FFMPEG")) or shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return _existing(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 - not installed / no binary for this platform
        return None


def find_ffprobe(explicit: Optional[str] = None) -> Optional[str]:
    return _existing(explicit) or _existing(os.environ.get("DVSA_FFPROBE")) or shutil.which("ffprobe")
