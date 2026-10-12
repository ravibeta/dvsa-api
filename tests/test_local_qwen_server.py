"""The local Qwen (Ollama) server follows the dvsa-api process, not the Azure agent."""

import sys
from types import SimpleNamespace

import pytest
from django.test import override_settings

from core.azure import local_qwen_server as lq


@pytest.fixture(autouse=True)
def _reset():
    lq._process, lq._host = None, None
    yield
    lq._process, lq._host = None, None


class _Proc:
    pid = 4242
    _handle = 0

    def __init__(self):
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def _patch_start(monkeypatch, popen_calls):
    monkeypatch.setattr(lq, "_find_ollama", lambda: "ollama")
    monkeypatch.setattr(lq, "_free_port", lambda: 51234)
    monkeypatch.setattr(lq, "_bind_lifetime_to_this_process", lambda p: None)
    monkeypatch.setattr(lq, "_wait_ready", lambda host, proc: True)
    monkeypatch.setattr(lq.atexit, "register", lambda f: None)

    def popen(cmd, **kw):
        popen_calls.append((cmd, kw))
        return _Proc()

    monkeypatch.setattr(lq.subprocess, "Popen", popen)


@override_settings(DVSA_QWEN_ENABLED="true")
def test_enabled_starts_ollama_on_its_own_port_and_points_the_settings_at_it(monkeypatch):
    from django.conf import settings

    calls = []
    _patch_start(monkeypatch, calls)
    monkeypatch.delenv("DVSA_OLLAMA_HOST", raising=False)
    assert lq.start_if_enabled() == "http://127.0.0.1:51234"
    (cmd, kw), = calls
    assert cmd == ["ollama", "serve"] and kw["env"]["OLLAMA_HOST"] == "127.0.0.1:51234"
    assert settings.DVSA_OLLAMA_HOST == "http://127.0.0.1:51234"
    # idempotent: a second call does not start another server
    assert lq.start_if_enabled() == "http://127.0.0.1:51234" and len(calls) == 1


@pytest.mark.parametrize("flag", ["false", "0", "off", "no"])
def test_disabled_flag_starts_nothing(monkeypatch, flag):
    calls = []
    _patch_start(monkeypatch, calls)
    with override_settings(DVSA_QWEN_ENABLED=flag):
        assert lq.start_if_enabled() is None
    assert calls == []


@override_settings(DVSA_QWEN_ENABLED="true")
def test_missing_ollama_or_failed_startup_never_breaks_the_api(monkeypatch):
    calls = []
    _patch_start(monkeypatch, calls)
    monkeypatch.setattr(lq, "_find_ollama", lambda: None)
    assert lq.start_if_enabled() is None and calls == []

    monkeypatch.setattr(lq, "_find_ollama", lambda: "ollama")
    monkeypatch.setattr(lq, "_wait_ready", lambda host, proc: False)
    assert lq.start_if_enabled() is None and lq._process is None


@override_settings(DVSA_QWEN_ENABLED="true")
def test_stop_terminates_the_server_it_started(monkeypatch):
    calls = []
    _patch_start(monkeypatch, calls)
    lq.start_if_enabled()
    proc = lq._process
    lq.stop()
    assert proc.terminated and lq._process is None and lq.host() is None


def test_azure_qwen_agent_is_independent_of_the_local_server_flag():
    from core.azure.agents import FoundryAgents

    with override_settings(DVSA_QWEN_ENABLED="false"):
        agents = FoundryAgents(SimpleNamespace(project_endpoint="https://x", qwen_enabled=False))
        import core.azure.qwen_agent as qa

        qa_calls = []
        orig = qa.ask_qwen_agent
        qa.ask_qwen_agent = lambda *a, **k: qa_calls.append(1) or "azure answer"
        try:
            assert agents.run_qwen_agent("q", "5") == "azure answer"
        finally:
            qa.ask_qwen_agent = orig


def test_only_a_serving_runserver_process_autostarts(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("RUN_MAIN", raising=False)
    monkeypatch.setattr(sys, "argv", ["manage.py", "runserver", "127.0.0.1:8000", "--noreload"])
    assert lq.should_autostart_for_this_process() is True
    monkeypatch.setattr(sys, "argv", ["manage.py", "runserver"])          # autoreload parent
    assert lq.should_autostart_for_this_process() is False
    monkeypatch.setenv("RUN_MAIN", "true")                                 # autoreload child
    assert lq.should_autostart_for_this_process() is True
    monkeypatch.setattr(sys, "argv", ["manage.py", "migrate"])
    assert lq.should_autostart_for_this_process() is False
