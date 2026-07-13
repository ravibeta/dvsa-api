# Security Guidance — DVSA Agent Kits

The agent kits are designed to run in local, cloud, and event-triggered contexts.
This document sets the security expectations for all of them. **No credentials are
ever committed to source control** — every kit reads configuration from environment
variables and ships only `.env.example` files that document variable *names*.

## Secret management
- Provide secrets via environment variables or a secrets manager (Azure Key Vault,
  AWS Secrets Manager, GCP Secret Manager, or Kubernetes `Secret` objects).
- `.env.example` files list required names only; copy to `.env` locally and keep
  `.env` out of git (already covered by the repo `.gitignore`).
- In Kubernetes, mount secrets via `envFrom.secretRef` — never bake them into images
  or `ConfigMap`s. See `cloud_autonomous/k8s/agent-runner-deployment.yaml`.

## Signature verification
- **Slack**: every request is verified with `SLACK_SIGNING_SECRET` using the
  documented `v0` HMAC-SHA256 scheme plus a ±5-minute timestamp window to defeat
  replay (`integrations/adapters/slack_app/slack_trigger.py`).
- **Webhooks**: the generic receiver verifies an `X-DVSA-Signature: sha256=…` HMAC
  computed with `WEBHOOK_SECRET` and rejects unsigned/invalid requests.
- Comparisons use `hmac.compare_digest` (constant-time) to avoid timing oracles.

## Least-privilege storage access
- Grant storage adapters only the scope they need: write access to the run-output
  prefix, read access to the source-video prefix — nothing broader.
- Prefer per-agent identities (workload identity / IAM role / managed identity) over
  shared long-lived keys, so access can be revoked per agent.
- Scope buckets/containers per environment (dev/stage/prod); never share a
  production bucket with a development runner.

## Token rotation & short-lived credentials
- Prefer short-lived, automatically rotated credentials (OIDC federation for GitHub
  Actions, managed identities in Azure, IRSA in EKS) over static API keys.
- Where static keys are unavoidable (`DVSA_API_KEY`), rotate on a schedule and on
  suspected compromise; the runner reads the key fresh from the environment so a
  rolling restart picks up a rotated value with no code change.
- Keep credential TTLs short enough that a leaked token expires before it is useful.

## PII redaction
- Drone footage may contain faces, licence plates, and location metadata. Treat
  frames and detections as potentially sensitive.
- Redact or blur PII before persisting derived artefacts; store only the minimum
  detection metadata required downstream.
- Keep structured logs free of raw imagery and precise coordinates — the kit logger
  emits identifiers (`run_id`, `trace_id`) and counts, not payloads.
- Apply retention limits to stored outputs and honour deletion requests.

## Reporting
Report suspected vulnerabilities privately to the repository maintainers rather than
opening a public issue.

See also `common/README_SECURITY_SNIPPETS.md` for copy-paste-ready verification and
redaction snippets.
