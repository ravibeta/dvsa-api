"""Kit 2 — Cloud Autonomous runner (FastAPI).

Serves unattended drone-video processing:

* ``POST /run``   — validate a :class:`RunInput`, execute the pipeline with retries,
  return a :class:`RunOutput`.
* ``GET /metrics`` — Prometheus-compatible text exposition of runner counters.
* ``GET /healthz`` — liveness; flips to 503 while draining for graceful shutdown.
* ``GET /readyz``  — readiness.

The framework layer is thin: the retry/metrics/execution **core is
framework-independent** (``execute_run``, ``RunnerMetrics``) so it is unit-testable
without FastAPI installed. FastAPI itself is imported lazily so the module stays
importable in minimal environments; the HTTP app is built by :func:`create_app`.
"""

# NOTE: this module deliberately does NOT use ``from __future__ import annotations``.
# FastAPI resolves endpoint parameter annotations at route-registration time; with
# stringized (PEP 563) annotations it cannot resolve the FastAPI types imported
# locally inside ``create_app`` (e.g. ``Request``), so it would misclassify the body.
# Eager annotations keep the runner's HTTP contract correct.

import os
import signal
import threading
import time
from typing import Any, Callable, Optional

from ..common import (
    LocalFileStorageAdapter,
    MockInferenceAdapter,
    RunInput,
    RunLogger,
    RunOutput,
)
from ..common.pipeline import run_pipeline

MAX_PARALLELISM = int(os.environ.get("RUNNER_MAX_PARALLELISM", "4"))
RETRY_ATTEMPTS = int(os.environ.get("RUNNER_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_S = float(os.environ.get("RUNNER_RETRY_BACKOFF_S", "0.05"))


# --------------------------------------------------------------------------- #
# Metrics (Prometheus-compatible, dependency-free).
# --------------------------------------------------------------------------- #
class RunnerMetrics:
    """Thread-safe counters/gauges rendered in Prometheus text format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.runs_total = 0
        self.runs_succeeded = 0
        self.runs_failed = 0
        self.retries_total = 0
        self.in_flight = 0
        self.latency_seconds_sum = 0.0

    def start(self) -> None:
        with self._lock:
            self.runs_total += 1
            self.in_flight += 1

    def finish(self, *, ok: bool, latency_s: float, retries: int) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            self.retries_total += retries
            self.latency_seconds_sum += latency_s
            if ok:
                self.runs_succeeded += 1
            else:
                self.runs_failed += 1

    def render(self) -> str:
        with self._lock:
            rows = [
                ("dvsa_runner_runs_total", "counter", self.runs_total),
                ("dvsa_runner_runs_succeeded_total", "counter", self.runs_succeeded),
                ("dvsa_runner_runs_failed_total", "counter", self.runs_failed),
                ("dvsa_runner_retries_total", "counter", self.retries_total),
                ("dvsa_runner_in_flight", "gauge", self.in_flight),
                ("dvsa_runner_latency_seconds_sum", "counter",
                 round(self.latency_seconds_sum, 6)),
            ]
        lines = []
        for name, kind, value in rows:
            lines.append(f"# TYPE {name} {kind}")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Execution core (no web framework required).
# --------------------------------------------------------------------------- #
def _with_retries(
    fn: Callable[[], Any], *, attempts: int, backoff_s: float,
) -> Any:
    """Call ``fn`` up to ``attempts`` times with linear backoff; return (result, tries)."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(), attempt - 1
        except Exception as exc:  # noqa: BLE001 - retry any pipeline error
            last_exc = exc
            if attempt < attempts:
                time.sleep(backoff_s * attempt)
    raise last_exc  # type: ignore[misc]


def _store_dest() -> Optional[str]:
    bucket = os.environ.get("STORAGE_BUCKET")
    if not bucket:
        return None
    # Offline default: treat non-remote buckets as a local directory.
    return bucket if "://" not in bucket or bucket.startswith("file://") else None


def execute_run(
    run_input: RunInput,
    *,
    metrics: Optional[RunnerMetrics] = None,
    inference: Optional[MockInferenceAdapter] = None,
    store_dest: Optional[str] = None,
) -> RunOutput:
    """Execute one run with retries and metrics; framework-independent."""
    metrics = metrics or _METRICS
    inference = inference or MockInferenceAdapter(
        os.environ.get("MODEL_VERSION", "mock-detector-1.0.0"))
    dest = store_dest if store_dest is not None else _store_dest()
    logger = RunLogger(agent_id=run_input.agent_id or "cloud-runner",
                       model_version=inference.model_version)
    metrics.start()
    started = time.time()
    ok = False
    retries = 0
    try:
        result, retries = _with_retries(
            lambda: run_pipeline(run_input, inference=inference,
                                 storage=LocalFileStorageAdapter(),
                                 store_dest=dest, logger=logger),
            attempts=RETRY_ATTEMPTS, backoff_s=RETRY_BACKOFF_S)
        ok = True
        return result
    finally:
        metrics.finish(ok=ok, latency_s=time.time() - started, retries=retries)


# Module-level singletons shared by the app.
_METRICS = RunnerMetrics()
_SEMAPHORE = threading.BoundedSemaphore(MAX_PARALLELISM)


class _State:
    """Tracks draining status for graceful shutdown."""

    draining = False


def request_shutdown(*_: Any) -> None:
    """Begin draining: stop reporting healthy so orchestrators divert traffic."""
    _State.draining = True


# --------------------------------------------------------------------------- #
# FastAPI application (lazy import — optional dependency).
# --------------------------------------------------------------------------- #
def create_app() -> Any:
    """Build and return the FastAPI app. Requires ``fastapi`` to be installed."""
    from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: PLC0415
    from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: PLC0415
    from starlette.concurrency import run_in_threadpool  # noqa: PLC0415

    def require_api_key(authorization: str = Header(default="")) -> None:
        expected = os.environ.get("DVSA_API_KEY")
        if not expected:  # auth disabled for local/dev when unset
            return
        import hmac  # noqa: PLC0415
        token = authorization[len("Bearer "):].strip() \
            if authorization.startswith("Bearer ") else authorization.strip()
        if not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    app = FastAPI(title="DVSA Cloud Autonomous Runner", version="0.1.0")

    # The body is parsed and validated through the shared ``RunInput`` schema rather
    # than a FastAPI-declared model: the module uses ``from __future__ import
    # annotations``, which turns declared model params into string forward-refs that
    # FastAPI cannot resolve for a locally-defined class.
    @app.post("/run")
    async def run(request: Request,
                  _auth: None = Depends(require_api_key)) -> JSONResponse:
        if _State.draining:
            raise HTTPException(status_code=503, detail="server draining")
        acquired = _SEMAPHORE.acquire(timeout=30)
        if not acquired:
            raise HTTPException(status_code=429, detail="runner at capacity")
        try:
            try:
                payload = await request.json()
                run_input = RunInput.model_validate(payload)
            except Exception as exc:  # noqa: BLE001 - malformed/invalid body
                raise HTTPException(status_code=422, detail=f"invalid request: {exc}")
            try:
                output = await run_in_threadpool(execute_run, run_input)
            except Exception as exc:  # noqa: BLE001 - surface as 500 with a message
                raise HTTPException(status_code=500, detail=str(exc))
            return JSONResponse(status_code=200, content=output.model_dump())
        finally:
            _SEMAPHORE.release()

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(_METRICS.render())

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        if _State.draining:
            return JSONResponse(status_code=503, content={"status": "draining"})
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.on_event("startup")
    def _register_signals() -> None:  # pragma: no cover - exercised in container
        try:
            signal.signal(signal.SIGTERM, request_shutdown)
        except (ValueError, OSError):
            pass  # not on the main thread (e.g. under TestClient) — that's fine

    return app


try:  # Build the module-level app when FastAPI is available (uvicorn entrypoint).
    app = create_app()
except Exception:  # noqa: BLE001 - fastapi not installed / import-time issue
    app = None


if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # noqa: PLC0415
    uvicorn.run("agent_kits.cloud_autonomous.cloud_runner:app",
                host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
