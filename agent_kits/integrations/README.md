# Kit 4 — Integrations & Triggers

**Purpose.** Launch drone-video analysis workflows from external systems — GitHub
Actions, Slack slash commands, or any signed webhook — converting each event into a
`RunInput` and tracking the resulting run.

## Architecture

```
  GitHub Actions ─┐
  Slack /command ─┼─▶ verify signature ─▶ payload → RunInput ─▶ dispatcher ─▶ pipeline
  Generic webhook ┘        (HMAC)             (validated)          │
                                                                   ▼
                                                             RunTracker
                                                     (accepted→running→completed)
```

Every trigger shares the same **pure-function core** (verify → convert → dispatch),
so it runs under any web framework or none. The default dispatcher runs the pipeline
in-process; swap in one that calls the Cloud Autonomous runner or a queue.

## Triggers

### GitHub Actions
[`adapters/github_actions.yaml`](adapters/github_actions.yaml) — a reference workflow
(kept out of `.github/workflows/` so it does not auto-run). It triggers on
`workflow_dispatch` or a `repository_dispatch` (`dvsa-run`) event and POSTs a run to
the cloud runner. Uses GitHub secrets + OIDC — no credentials in the file.

### Slack slash command
`/dvsa-run <video_uri> [fps=5 sensor_id=cam1]`

- [`adapters/slack_app/manifest.json`](adapters/slack_app/manifest.json) — app manifest.
- [`adapters/slack_app/slack_trigger.py`](adapters/slack_app/slack_trigger.py) —
  `handle_slack_command(headers, body)`; verifies the Slack `v0` signature (±5-min
  replay window) and parses the command into a `RunInput`.

```python
from agent_kits.integrations.adapters.slack_app.slack_trigger import handle_slack_command
status, slack_json = handle_slack_command(request.headers, request.body)
```

### Generic webhook
[`adapters/webhook_receiver.py`](adapters/webhook_receiver.py) —
`handle_webhook(headers, body)`; verifies `X-DVSA-Signature: sha256=…` (HMAC with
`WEBHOOK_SECRET`) and converts the JSON payload. A dependency-free stdlib server is
available via `make_server()`:

```bash
WEBHOOK_SECRET=… python -c \
  "from agent_kits.integrations.adapters.webhook_receiver import make_server; \
   make_server(port=8090).serve_forever()"
```

## Quickstart (verify a signed call, offline)

```python
import json
from agent_kits.integrations.adapters.webhook_receiver import compute_signature, handle_webhook

body = json.dumps({"video_uri": "file://…/sample_short.json",
                   "processing_flags": {"fps": 5}}).encode()
headers = {"X-DVSA-Signature": compute_signature(body, "my-secret")}
status, resp = handle_webhook(headers, body, secret="my-secret")
# 202, resp["run"]["status"] == "completed"
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `SLACK_SIGNING_SECRET` | HMAC secret for verifying Slack requests. |
| `WEBHOOK_SECRET` | HMAC secret for verifying generic webhooks. |
| `DVSA_RUNNER_URL` / `DVSA_API_KEY` | Cloud runner target for the GitHub Actions trigger (GH secrets). |

**No secrets are committed.** Verification fails closed when a secret is unset.

## Sample output

```json
{"accepted": true,
 "run": {"run_id": "trig_…", "status": "completed", "detections": 33,
         "history": ["accepted", "running", "completed"]}}
```

## Cost control

- Validate/authenticate before dispatching so unsigned events never spend inference.
- Point the dispatcher at the cloud runner (with its parallelism cap) rather than
  running heavy jobs inline in the receiver.

## Extension guide

- **New trigger**: reuse `run_tracker.make_default_dispatcher` and the verify/convert
  pattern; add a `*_to_run_input` converter for your payload shape.
- **Async dispatch**: replace the default dispatcher with one that enqueues to
  Celery/SQS/PubSub and returns immediately with `accepted`.

## Security

Signature verification uses `hmac.compare_digest` (constant-time); Slack requests are
additionally protected by a ±5-minute replay window. See `agent_kits/SECURITY.md` and
`common/README_SECURITY_SNIPPETS.md`.
