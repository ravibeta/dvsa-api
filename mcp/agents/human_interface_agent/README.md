# `human_interface_agent`

Human-in-the-loop confirmation gate (capability `human_interface`). In production
`handle_task` posts to a webhook/UI and returns `status: "in_progress"` until a
person responds. For offline tests/sims it auto-confirms when
`MCP_HUMAN_AUTO_CONFIRM` is set (default `1`), returning an `approved` decision;
set it to `0` to require a real out-of-band response. No extra dependencies.
