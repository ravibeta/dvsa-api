"""Lifecycle of the *local* Qwen (Ollama) server owned by the dvsa-api process.

``DVSA_QWEN_ENABLED`` is checked once, at API startup. When it is on, dvsa-api
starts its own ``ollama serve`` on a free loopback port (or
``DVSA_LOCAL_QWEN_PORT``), points the local-Qwen settings at it, and stops it
when the API process exits -- on Windows even if the API is force-killed, via a
kill-on-close job object. The port is chosen by dvsa-api, so any other Ollama /
Qwen server on the machine keeps running independently on its own port; dvsa-api
only ever talks to the one it started.

The Azure Foundry ``qwen-agent-in-a-team`` is a separate path and is never
affected by this flag.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Optional

logger = logging.getLogger("apps.azure")

_STARTUP_TIMEOUT_SECONDS = 20

_process: Optional[subprocess.Popen] = None
_job_handle = None  # Windows job object; closing it kills the child
_host: Optional[str] = None


def _truthy(value) -> bool:
    return str(value).strip().lower() not in ("", "0", "false", "no", "off", "none")


def _find_ollama() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = os.path.join(local, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(candidate):
            return candidate
    return None


def _free_port() -> int:
    fixed = os.environ.get("DVSA_LOCAL_QWEN_PORT")
    if fixed:
        return int(fixed)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bind_lifetime_to_this_process(proc: subprocess.Popen) -> None:
    """Windows: make the OS kill ``proc`` when this process dies, however it dies."""
    global _job_handle
    if os.name != "nt":
        return
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        # 9 == JobObjectExtendedLimitInformation
        kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))  # noqa: SLF001
        _job_handle = job  # keep open for the lifetime of this process
    except Exception as exc:  # noqa: BLE001
        logger.warning("local qwen: could not bind server lifetime to this process: %s", exc)


def _wait_ready(host: str, proc: subprocess.Popen) -> bool:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            # Loopback must never go through a configured HTTP(S) proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"{host}/api/version", timeout=1):
                return True
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    return False


def host() -> Optional[str]:
    """URL of the server this process started, or None."""
    return _host


def start_if_enabled() -> Optional[str]:
    """Start the local Qwen server when ``DVSA_QWEN_ENABLED`` is on. Idempotent.

    Returns the server URL, or None when disabled / Ollama is unavailable (the
    API keeps running; local Qwen calls then fall back to "No comment").
    """
    global _process, _host
    from django.conf import settings  # noqa: PLC0415

    if _process is not None and _process.poll() is None:
        return _host
    if not _truthy(getattr(settings, "DVSA_QWEN_ENABLED", "true")):
        return None
    exe = _find_ollama()
    if not exe:
        logger.info("local qwen: DVSA_QWEN_ENABLED is on but the ollama executable was not found")
        return None

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}"}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.Popen([exe, "serve"], env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=flags)
    except OSError as exc:
        logger.warning("local qwen: could not start ollama: %s", exc)
        return None
    _bind_lifetime_to_this_process(proc)
    if not _wait_ready(url, proc):
        logger.warning("local qwen: server did not become ready on %s; stopping it", url)
        _terminate(proc)
        return None

    _process, _host = proc, url
    no_proxy = {h for h in os.environ.get("NO_PROXY", "").split(",") if h}
    os.environ["NO_PROXY"] = ",".join(sorted(no_proxy | {"127.0.0.1", "localhost"}))
    # The baseline / local-Qwen clients read the host from settings or the env.
    settings.DVSA_OLLAMA_HOST = url
    os.environ["DVSA_OLLAMA_HOST"] = url
    atexit.register(stop)
    logger.info("local qwen: started ollama on %s (pid %s); stops with this process", url, proc.pid)
    return url


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def stop() -> None:
    global _process, _host
    if _process is not None:
        _terminate(_process)
        logger.info("local qwen: stopped ollama on %s", _host)
    _process, _host = None, None


def should_autostart_for_this_process() -> bool:
    """True only for a process that is actually serving requests.

    Not for tests, migrations, shells or other management commands, and not for
    the autoreloader's watcher parent (only the serving child starts it).
    """
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    argv = sys.argv
    if len(argv) > 1 and argv[1] == "runserver":
        return "--noreload" in argv or os.environ.get("RUN_MAIN") == "true"
    return False
