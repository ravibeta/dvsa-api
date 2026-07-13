# Cloud Autonomous runner image.
#
# Builds a minimal container that serves the FastAPI runner with uvicorn. Only the
# agent-kit dependencies are installed (not the full Django app) to keep the image
# small. No secrets are baked in — configuration is provided at runtime via env vars
# / mounted secrets (see k8s/agent-runner-deployment.yaml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    RUNNER_MAX_PARALLELISM=4

WORKDIR /app

# Install runtime deps first for better layer caching.
COPY agent_kits/requirements.txt /app/agent_kits/requirements.txt
RUN pip install --no-cache-dir -r /app/agent_kits/requirements.txt uvicorn==0.30.6

# Copy only what the runner needs.
COPY agent_kits /app/agent_kits

# Run as an unprivileged user (least privilege).
RUN useradd --create-home --uid 10001 runner && chown -R runner /app
USER runner

EXPOSE 8080

# Graceful shutdown: uvicorn handles SIGTERM; the app drains in-flight requests.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"

CMD ["uvicorn", "agent_kits.cloud_autonomous.cloud_runner:app", \
     "--host", "0.0.0.0", "--port", "8080"]
