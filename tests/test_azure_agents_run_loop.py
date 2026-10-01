"""Unit tests for ``FoundryAgents._run_agent``'s tool-call/poll loop.

Live symptom this guards against: asking a question in the console ("how
many land bridges...") never returned an answer, and the server log grew
forever with the same failing ``ask_perplexity`` call repeating every ~1s.

Root cause: when every tool_call in a ``requires_action`` batch either raised
or legitimately returned ``None`` (e.g. ``ask_perplexity`` returning ``None``
on an unresolvable account/video), no ``ToolOutput`` was appended for it, so
``tool_outputs`` stayed empty and ``submit_tool_outputs`` was never called.
The run's ``required_action`` was therefore never satisfied, so the *same*
pending tool_call kept coming back on every poll and the loop re-executed it
forever. ``_run_agent`` now always submits a ToolOutput per tool_call
(empty/error string when the tool gave nothing), and a hard wall-clock cap
(``_AGENT_MAX_RUN_SECONDS``) guards against any other way a run could fail to
settle.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from core.azure.agents import FoundryAgents


class _FakeSubmitToolOutputsAction:
    """Stand-in for azure.ai.agents.models.SubmitToolOutputsAction — only
    _run_agent's isinstance() check and .submit_tool_outputs.tool_calls
    attribute access need to work."""

    def __init__(self, tool_calls):
        self.submit_tool_outputs = SimpleNamespace(tool_calls=tool_calls)


def _patch_action_class(monkeypatch):
    import azure.ai.agents.models as agents_models

    monkeypatch.setattr(agents_models, "SubmitToolOutputsAction", _FakeSubmitToolOutputsAction)


def test_failing_tool_call_still_advances_the_run(monkeypatch):
    """A tool_call whose executor returns None must still get a ToolOutput
    submitted, so the run can move past requires_action instead of looping."""
    _patch_action_class(monkeypatch)
    agents = FoundryAgents(SimpleNamespace())

    tool_call = SimpleNamespace(id="call-1")
    pending_run = SimpleNamespace(
        id="run-1", status="requires_action",
        required_action=_FakeSubmitToolOutputsAction(tool_calls=[tool_call]),
    )
    done_run = SimpleNamespace(id="run-1", status="completed", required_action=None)

    fake_client = MagicMock()
    fake_client.threads.create.return_value = SimpleNamespace(id="thread-1")
    fake_client.runs.create.return_value = pending_run
    # First poll still pending (forces one requires_action round), second poll done.
    fake_client.runs.get.side_effect = [pending_run, done_run]
    fake_client.messages.list.return_value = []

    monkeypatch.setattr("core.azure.agents.time.sleep", lambda *_: None)

    executor = MagicMock(return_value=None)  # the tool legitimately gives nothing
    agents._run_agent(fake_client, SimpleNamespace(id="agent-1"), "hello", executor)

    # submit_tool_outputs must be called even though the tool returned None —
    # this is the fix: previously it was skipped entirely in this case.
    fake_client.runs.submit_tool_outputs.assert_called_once()
    submitted = fake_client.runs.submit_tool_outputs.call_args.kwargs["tool_outputs"]
    assert len(submitted) == 1
    assert submitted[0].tool_call_id == "call-1"
    # Only two runs.get() polls happened — proof the loop didn't spin forever.
    assert fake_client.runs.get.call_count == 2


def test_raising_tool_call_still_advances_the_run(monkeypatch):
    _patch_action_class(monkeypatch)
    agents = FoundryAgents(SimpleNamespace())

    tool_call = SimpleNamespace(id="call-1")
    pending_run = SimpleNamespace(
        id="run-1", status="requires_action",
        required_action=_FakeSubmitToolOutputsAction(tool_calls=[tool_call]),
    )
    done_run = SimpleNamespace(id="run-1", status="completed", required_action=None)

    fake_client = MagicMock()
    fake_client.threads.create.return_value = SimpleNamespace(id="thread-1")
    fake_client.runs.create.return_value = pending_run
    fake_client.runs.get.side_effect = [pending_run, done_run]
    fake_client.messages.list.return_value = []

    monkeypatch.setattr("core.azure.agents.time.sleep", lambda *_: None)

    def boom(_call):
        raise RuntimeError("tool blew up")

    agents._run_agent(fake_client, SimpleNamespace(id="agent-1"), "hello", boom)

    fake_client.runs.submit_tool_outputs.assert_called_once()
    submitted = fake_client.runs.submit_tool_outputs.call_args.kwargs["tool_outputs"]
    assert "tool blew up" in submitted[0].output


def test_tool_round_cap_cancels_a_model_stuck_retrying(monkeypatch):
    """Live symptom: the model kept calling describe_frame with a new wrong
    kwarg each round (frame_index, then scene_img, ...), never converging.
    The SDK's FunctionTool.execute() already catches that TypeError and
    returns {"error": "..."} rather than raising, so submit_tool_outputs was
    always being called — the wall-clock deadline was the only backstop, and
    at ~1-2s of model latency per round it could burn most of
    _AGENT_MAX_RUN_SECONDS before firing. The round cap cuts this off fast,
    independent of wall clock."""
    _patch_action_class(monkeypatch)
    monkeypatch.setattr("core.azure.agents._AGENT_MAX_TOOL_ROUNDS", 2)
    agents = FoundryAgents(SimpleNamespace())

    def make_pending():
        tool_call = SimpleNamespace(id="call-x", function=SimpleNamespace(name="describe_frame"))
        return SimpleNamespace(
            id="run-1", status="requires_action",
            required_action=_FakeSubmitToolOutputsAction(tool_calls=[tool_call]),
        )

    fake_client = MagicMock()
    fake_client.threads.create.return_value = SimpleNamespace(id="thread-1")
    fake_client.runs.create.return_value = make_pending()
    # Keeps reporting a fresh requires_action every poll, forever, if nothing
    # stops the loop.
    fake_client.runs.get.side_effect = lambda **_: make_pending()
    fake_client.messages.list.return_value = []

    monkeypatch.setattr("core.azure.agents.time.sleep", lambda *_: None)

    executor = MagicMock(return_value='{"error": "describe_frame() got an unexpected keyword argument \'x\'"}')
    agents._run_agent(fake_client, SimpleNamespace(id="agent-1"), "hello", executor)

    fake_client.runs.cancel.assert_called_once_with(thread_id="thread-1", run_id="run-1")
    # _AGENT_MAX_TOOL_ROUNDS (2) rounds were actually processed/submitted;
    # the 3rd poll is what detects the cap was exceeded and cancels — proof
    # this terminates in a handful of rounds, not up to the full wall-clock
    # budget.
    assert fake_client.runs.submit_tool_outputs.call_count == 2
    assert fake_client.runs.get.call_count == 3


def test_repeated_failures_of_same_function_trigger_a_stop_message(monkeypatch):
    _patch_action_class(monkeypatch)
    monkeypatch.setattr("core.azure.agents._AGENT_MAX_TOOL_ROUNDS", 10)
    agents = FoundryAgents(SimpleNamespace())

    def make_pending(call_id):
        tool_call = SimpleNamespace(id=call_id, function=SimpleNamespace(name="describe_frame"))
        return SimpleNamespace(
            id="run-1", status="requires_action",
            required_action=_FakeSubmitToolOutputsAction(tool_calls=[tool_call]),
        )

    done_run = SimpleNamespace(id="run-1", status="completed", required_action=None)

    fake_client = MagicMock()
    fake_client.threads.create.return_value = SimpleNamespace(id="thread-1")
    fake_client.runs.create.return_value = make_pending("call-1")
    fake_client.runs.get.side_effect = [make_pending("call-1"), make_pending("call-2"), done_run]
    fake_client.messages.list.return_value = []

    monkeypatch.setattr("core.azure.agents.time.sleep", lambda *_: None)

    executor = MagicMock(return_value='{"error": "unexpected keyword argument"}')
    agents._run_agent(fake_client, SimpleNamespace(id="agent-1"), "hello", executor)

    calls = fake_client.runs.submit_tool_outputs.call_args_list
    assert len(calls) == 2
    first_output = calls[0].kwargs["tool_outputs"][0].output
    second_output = calls[1].kwargs["tool_outputs"][0].output
    assert "Stop calling it" not in first_output
    assert "Stop calling it" in second_output


def test_run_that_never_settles_is_cancelled_after_the_deadline(monkeypatch):
    """Defense in depth: even if something else keeps a run in
    requires_action/in_progress forever, _run_agent must give up and return
    rather than hang the chat request indefinitely."""
    _patch_action_class(monkeypatch)
    monkeypatch.setattr("core.azure.agents._AGENT_MAX_RUN_SECONDS", -1)
    agents = FoundryAgents(SimpleNamespace())

    tool_call = SimpleNamespace(id="call-1")
    pending_run = SimpleNamespace(
        id="run-1", status="requires_action",
        required_action=_FakeSubmitToolOutputsAction(tool_calls=[tool_call]),
    )

    fake_client = MagicMock()
    fake_client.threads.create.return_value = SimpleNamespace(id="thread-1")
    fake_client.runs.create.return_value = pending_run
    fake_client.runs.get.return_value = pending_run  # never settles
    fake_client.messages.list.return_value = []

    monkeypatch.setattr("core.azure.agents.time.sleep", lambda *_: None)

    agents._run_agent(fake_client, SimpleNamespace(id="agent-1"), "hello", MagicMock(return_value=None))

    # The deadline (0s) is already past on the first check -> cancelled
    # immediately, no poll ever happens.
    fake_client.runs.cancel.assert_called_once_with(thread_id="thread-1", run_id="run-1")
    fake_client.runs.get.assert_not_called()
