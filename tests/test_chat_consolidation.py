"""The console's Ask must return one consolidated answer, never the raw bundle.

Live symptom: asking "How many land bridges are encountered?" showed
``[User]: ... [Knowledge Search Agent Output]: None [Connected Function Agent
Output]: ...`` -- the team's raw outputs concatenated -- instead of the
consolidator's narrative. Cause: ``_run_agent`` returned the last text message
in the thread, which is the *user* message we posted whenever the consolidator
run was cancelled / failed / out of budget, and ``synthesize_from_chat_agent``
also fell back to the raw bundle in those cases.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.azure.agents import FoundryAgents, _source_text


def _msg(role, text):
    return SimpleNamespace(role=role, text_messages=[SimpleNamespace(text=SimpleNamespace(value=text))])


def _run_agent_with_messages(messages, status="completed"):
    client = MagicMock()
    client.threads.create.return_value = SimpleNamespace(id="t1")
    run = SimpleNamespace(id="r1", status=status, required_action=None)
    client.runs.create.return_value = run
    client.messages.list.return_value = messages
    return FoundryAgents(SimpleNamespace())._run_agent(client, SimpleNamespace(id="a1"), "hello", None)


def test_run_agent_never_returns_the_user_message_as_the_answer():
    assert _run_agent_with_messages([_msg("user", "[User]: hi ...")], status="cancelled") is None


def test_run_agent_returns_the_assistant_reply():
    msgs = [_msg("user", "question"), _msg("assistant", "the answer")]
    assert _run_agent_with_messages(msgs) == "the answer"


def test_run_agent_handles_enum_style_roles():
    msgs = [_msg(SimpleNamespace(value="user"), "q"), _msg(SimpleNamespace(value="assistant"), "a")]
    assert _run_agent_with_messages(msgs) == "a"


def test_source_text_treats_none_and_blank_as_missing():
    for v in (None, "", "  ", "None", "null"):
        assert _source_text(v) is None
    assert _source_text(" two bridges ") == "two bridges"


# --- synthesize_from_chat_agent ---------------------------------------------
def _team(monkeypatch, search=None, functions="F out", analyzer="A out", qwen=None, configured=True):
    agents = FoundryAgents(SimpleNamespace(chat_agent_name="chat-agent-in-a-team", agent_model="m"))
    monkeypatch.setattr(type(agents), "configured", property(lambda self: configured))
    monkeypatch.setattr(agents, "run_connected_agent", lambda *a, **k: search)
    monkeypatch.setattr(agents, "run_function_tools", lambda *a, **k: functions)
    monkeypatch.setattr(agents, "run_analyzer_tools", lambda *a, **k: analyzer)
    monkeypatch.setattr(agents, "run_qwen_agent", lambda *a, **k: qwen)
    monkeypatch.setattr(agents, "_agents_client", lambda: nullcontext(MagicMock()))
    monkeypatch.setattr(agents, "_find_agent", lambda client, name: SimpleNamespace(id="chat"))
    return agents


def test_consolidator_gets_every_source_and_marks_missing_ones_unavailable(monkeypatch):
    agents = _team(monkeypatch, search=None, functions="FUNC two bridges", analyzer="TOOL four tracks", qwen="QWEN ok")
    seen = {}
    monkeypatch.setattr(agents, "_run_agent",
                        lambda client, agent, content, ex, deadline=None: seen.update(content=content) or "Final story.")
    assert agents.synthesize_from_chat_agent("How many land bridges?", "5", "2") == "Final story."
    c = seen["content"]
    for token in ("FUNC two bridges", "TOOL four tracks", "QWEN ok", "How many land bridges?"):
        assert token in c
    assert "[Knowledge Search Agent Output]:\nunavailable" in c       # not the literal "None"
    assert "Output]:\nNone" not in c
    assert "faithful" in c                                            # fidelity instruction travels with the data
    assert "NEVER mention agents" in c                                # one unified voice, no agent attribution


def test_failed_consolidation_returns_readable_summary_not_the_raw_bundle(monkeypatch):
    agents = _team(monkeypatch, functions="No land bridges seen.", analyzer="Tracks and buildings.")
    monkeypatch.setattr(agents, "_run_agent", lambda *a, **k: None)
    out = agents.synthesize_from_chat_agent("How many land bridges?", "5", "2")
    assert not out.startswith("[User]:") and "[Knowledge Search Agent Output]" not in out
    assert "No land bridges seen." in out and "Tracks and buildings." in out


def test_echoed_user_bundle_is_never_returned(monkeypatch):
    agents = _team(monkeypatch)
    monkeypatch.setattr(agents, "_run_agent", lambda *a, **k: "[User]: q\n\n[Tool Agent Output]: x")
    assert not agents.synthesize_from_chat_agent("q", "5", "2").startswith("[User]:")


def test_exhausted_budget_returns_readable_summary(monkeypatch):
    agents = _team(monkeypatch)
    monkeypatch.setattr("core.azure.agents._CHAT_TURN_MAX_SECONDS", -1)
    called = []
    monkeypatch.setattr(agents, "_run_agent", lambda *a, **k: called.append(1) or "x")
    out = agents.synthesize_from_chat_agent("q", "5", "2")
    # Team members were skipped too, so nothing to report -- but never the raw bundle.
    assert called == [] and not out.startswith("[User]:") and "try again" in out


def test_team_members_cannot_spend_the_consolidators_reserved_budget(monkeypatch):
    import time

    agents = _team(monkeypatch)
    deadlines = {}
    monkeypatch.setattr(agents, "run_function_tools", lambda *a, **k: deadlines.update(team=k["deadline"]) or "F")
    monkeypatch.setattr(agents, "_run_agent",
                        lambda c, a, content, ex, deadline=None: deadlines.update(final=deadline) or "ok")
    agents.synthesize_from_chat_agent("q", "5", "2")
    assert deadlines["final"] - deadlines["team"] >= 19
    assert deadlines["team"] > time.monotonic()


# --- Qwen runs in parallel with the rest of the team ---------------------------
def test_qwen_runs_in_parallel_with_the_other_team_members(monkeypatch):
    import threading
    import time

    agents = _team(monkeypatch, qwen=None)
    qwen_started, others_saw_qwen_running = threading.Event(), []

    def qwen(*a, **k):
        qwen_started.set()
        time.sleep(0.3)
        return "QWEN parallel finding"

    def functions(*a, **k):
        # Only passes if Qwen was already started (not waiting for us to finish).
        others_saw_qwen_running.append(qwen_started.wait(timeout=2))
        return "F"

    monkeypatch.setattr(agents, "run_qwen_agent", qwen)
    monkeypatch.setattr(agents, "run_function_tools", functions)
    seen = {}
    monkeypatch.setattr(agents, "_run_agent",
                        lambda c, a, content, ex, deadline=None: seen.update(content=content) or "Final.")
    assert agents.synthesize_from_chat_agent("q", "5", "2") == "Final."
    assert others_saw_qwen_running == [True]
    assert "QWEN parallel finding" in seen["content"]


def test_qwen_failure_or_slowness_never_blocks_the_consolidated_answer(monkeypatch):
    import time

    agents = _team(monkeypatch)
    monkeypatch.setattr(agents, "run_qwen_agent", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    seen = {}
    monkeypatch.setattr(agents, "_run_agent",
                        lambda c, a, content, ex, deadline=None: seen.update(content=content) or "Final.")
    assert agents.synthesize_from_chat_agent("q", "5", "2") == "Final."
    assert "[Qwen Vision Agent Output]:\nunavailable" in seen["content"]

    # A Qwen call that outlives the team deadline is abandoned, not waited for.
    agents = _team(monkeypatch)
    monkeypatch.setattr("core.azure.agents._CONSOLIDATION_RESERVE_SECONDS", 0)
    monkeypatch.setattr("core.azure.agents._CHAT_TURN_MAX_SECONDS", 1)
    monkeypatch.setattr(agents, "run_qwen_agent", lambda *a, **k: time.sleep(3) or "LATE_QWEN_ANSWER")
    monkeypatch.setattr(agents, "_run_agent",
                        lambda c, a, content, ex, deadline=None: seen.update(content=content) or "Final.")
    start = time.monotonic()
    agents.synthesize_from_chat_agent("q", "5", "2")
    assert time.monotonic() - start < 2.5 and "LATE_QWEN_ANSWER" not in seen["content"]
