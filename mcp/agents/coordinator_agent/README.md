# `coordinator_agent`

Mission-level coordinator (capability `coordinator`). Aggregates upstream
analyzer results and decides the outcome. On a confirmed anomaly at/above the
escalation threshold (`config.escalation_confidence`, overridable by the
mission's `escalation_rules.min_confidence`) with `require_human`, it returns
`status: "delegated"` and emits a dynamic follow-up task for a `human_interface`
agent — high-impact actions require explicit human confirmation. No extra deps.
