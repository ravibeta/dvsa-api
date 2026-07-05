"""Planner tests — mission template -> validated task DAG.

Pure-Python: no Django, no network.
"""

import pytest

from dvsa_api.mcp.errors import MissionError
from dvsa_api.mcp.planner import plan_mission


def _template(**over):
    base = {
        "mission_id": "m1",
        "description": "test",
        "tasks": [
            {"id": "a", "capability": "scout"},
            {"id": "b", "capability": "analyzer", "depends_on": ["a"]},
            {"id": "c", "capability": "coordinator", "depends_on": ["b"]},
        ],
    }
    base.update(over)
    return base


def test_plan_builds_graph_with_defaults():
    mission = plan_mission(_template())
    assert mission.mission_id == "m1"
    assert len(mission.graph.nodes()) == 3
    # Only the root task is ready initially.
    ready = [n.task.task_id for n in mission.graph.ready_tasks()]
    assert ready == ["a"]


def test_ready_tasks_progress_with_completion():
    mission = plan_mission(_template())
    graph = mission.graph
    graph.node("a").status = "completed"
    assert [n.task.task_id for n in graph.ready_tasks()] == ["b"]
    graph.node("b").status = "completed"
    assert [n.task.task_id for n in graph.ready_tasks()] == ["c"]


def test_missing_tasks_raises():
    with pytest.raises(MissionError):
        plan_mission({"mission_id": "x", "tasks": []})


def test_task_without_capability_or_agent_raises():
    with pytest.raises(MissionError):
        plan_mission({"tasks": [{"id": "a"}]})


def test_unknown_dependency_raises():
    with pytest.raises(MissionError):
        plan_mission({"tasks": [
            {"id": "a", "capability": "scout", "depends_on": ["ghost"]}]})


def test_cycle_is_detected():
    with pytest.raises(MissionError):
        plan_mission({"tasks": [
            {"id": "a", "capability": "scout", "depends_on": ["b"]},
            {"id": "b", "capability": "scout", "depends_on": ["a"]},
        ]})


def test_retry_and_timeout_defaults_propagate():
    mission = plan_mission(_template(
        timeouts={"task_ms": 500}, retry_policy={"max_retries": 3}))
    node = mission.graph.node("a")
    assert node.task.deadline_ms == 500
    assert node.task.max_retries == 3
