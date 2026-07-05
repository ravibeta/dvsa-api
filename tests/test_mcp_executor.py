"""Executor tests — scheduling, retries, timeouts, follow-ups, concurrency.

Pure-Python: no Django, no network. Uses lightweight fake agents registered as
instances and referenced by ``agent`` (preferred_agent) in the mission template.
"""

import time

import pytest

from dvsa_api.mcp import registry
from dvsa_api.mcp.executor import Executor
from dvsa_api.mcp.message_bus import reset_bus
from dvsa_api.mcp.metrics import get_metrics
from dvsa_api.mcp.planner import plan_mission


def _ok_result(agent, extra=None):
    return {"status": "completed", "result": extra or {"ok": True},
            "metadata": {"agent": agent, "version": "1.0", "latency_ms": 1}}


class _OkAgent:
    def handle_task(self, task, context):
        return _ok_result("ok_agent")

    def health_check(self):
        return {"status": "ok"}


class _FailsOnceAgent:
    def __init__(self):
        self.calls = 0

    def handle_task(self, task, context):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient boom")
        return _ok_result("flaky_agent", {"recovered": True})

    def health_check(self):
        return {"status": "ok"}


class _SlowAgent:
    def handle_task(self, task, context):
        time.sleep(0.5)
        return _ok_result("slow_agent")

    def health_check(self):
        return {"status": "ok"}


class _FollowupAgent:
    def handle_task(self, task, context):
        return {
            "status": "completed",
            "result": {"followups": [
                {"id": "child", "type": "t", "agent": "ok_agent"}]},
            "metadata": {"agent": "followup_agent", "version": "1.0", "latency_ms": 1},
        }

    def health_check(self):
        return {"status": "ok"}


@pytest.fixture(autouse=True)
def _isolate():
    reset_bus()
    get_metrics().reset()
    registry.register_instance("ok_agent", _OkAgent())
    registry.register_instance("flaky_agent", _FailsOnceAgent())
    registry.register_instance("slow_agent", _SlowAgent())
    registry.register_instance("followup_agent", _FollowupAgent())
    yield


def _run(tasks, **over):
    template = {"mission_id": "t", "tasks": tasks}
    template.update(over)
    return Executor().run_mission(plan_mission(template), {})


def test_happy_path_chain_completes():
    summary = _run([
        {"id": "a", "agent": "ok_agent"},
        {"id": "b", "agent": "ok_agent", "depends_on": ["a"]},
    ])
    assert summary["status"] == "completed"
    assert all(t["status"] == "completed" for t in summary["tasks"])
    assert get_metrics().snapshot()["tasks_completed"] == 2


def test_retry_recovers_transient_failure():
    summary = _run(
        [{"id": "a", "agent": "flaky_agent", "max_retries": 1}])
    assert summary["status"] == "completed"
    assert summary["results"]["a"]["result"]["recovered"] is True


def test_timeout_marks_task_failed():
    summary = _run([
        {"id": "a", "agent": "slow_agent", "deadline_ms": 50, "max_retries": 0}])
    assert summary["status"] == "failed"
    node = next(t for t in summary["tasks"] if t["task_id"] == "a")
    assert node["status"] == "failed"
    assert get_metrics().snapshot()["task_failures"] >= 1


def test_dynamic_followup_is_scheduled_and_run():
    summary = _run([{"id": "root", "agent": "followup_agent"}])
    assert summary["status"] == "completed"
    ids = {t["task_id"] for t in summary["tasks"]}
    assert "child" in ids  # follow-up task was added and executed


def test_unknown_agent_capability_fails_gracefully():
    summary = _run([{"id": "a", "capability": "nonexistent_cap"}])
    assert summary["status"] == "failed"


def test_independent_tasks_both_complete():
    summary = _run([
        {"id": "a", "agent": "ok_agent"},
        {"id": "b", "agent": "ok_agent"},
    ])
    assert summary["status"] == "completed"
    assert len(summary["tasks"]) == 2
