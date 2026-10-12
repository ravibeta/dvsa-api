"""qwen-agent-in-a-team: frame selection, inline images, direct-deployment call."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.azure import qwen_agent as qa
from core.azure.agents import FoundryAgents

ENDPOINT = "https://found-vision-1.services.ai.azure.com/api/projects/droneimage"


def _jpeg_bytes(w=1280, h=720):
    import cv2

    img = np.full((h, w, 3), 90, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class Sess:
    def __init__(self, resp=None, content=None):
        self.resp, self.content, self.posts, self.gets = resp, content, [], []

    def get(self, url, **kw):
        self.gets.append(url)
        return SimpleNamespace(content=self.content, raise_for_status=lambda: None)

    def post(self, url, **kw):
        self.posts.append((url, kw))
        r = MagicMock()
        r.json.return_value = self.resp
        return r


# --- frame selection ---------------------------------------------------------
def test_select_frames_caps_at_four_evenly_spaced_keeping_ends():
    urls = [f"u{i}" for i in range(20)]
    picked = qa.select_frames(urls)
    assert len(picked) == 4 and picked[0] == "u0" and picked[-1] == "u19"
    assert picked == sorted(picked, key=lambda u: int(u[1:]))


def test_select_frames_small_and_empty_sets():
    assert qa.select_frames(["a", "b"]) == ["a", "b"]
    assert qa.select_frames([]) == []
    assert qa.select_frames(["a", "b", "c"], limit=1) == ["a"]


# --- inline images -----------------------------------------------------------
def test_inline_frame_downscales_and_returns_base64_data_url():
    import cv2

    url = qa.inline_frame("https://x/f.jpg?sig=1", session=Sess(content=_jpeg_bytes()))
    assert url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    h, w = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    assert max(h, w) == qa.MAX_SIDE


def test_inline_frame_rejects_non_images():
    with pytest.raises(ValueError):
        qa.inline_frame("https://x/f", session=Sess(content=b"not an image"))


# --- request building / parsing ---------------------------------------------
def test_build_messages_attaches_images_inline_with_instructions_as_system():
    m = qa.build_messages("how many bridges?", "SYS", ["data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB"])
    assert m[0] == {"role": "system", "content": "SYS"}
    parts = m[1]["content"]
    assert parts[0]["type"] == "text" and "2 frame(s)" in parts[0]["text"]
    assert [p["type"] for p in parts[1:]] == ["image_url", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_parse_completion_accepts_content_or_reasoning():
    assert qa.parse_completion({"choices": [{"message": {"content": " yes "}}]}) == "yes"
    assert qa.parse_completion({"choices": [{"message": {"content": None, "reasoning": "Yes"}}]}) == "Yes"
    assert qa.parse_completion({"choices": []}) is None


# --- end to end (mocked network) ---------------------------------------------
@pytest.fixture
def cfg():
    return SimpleNamespace(project_endpoint=ENDPOINT)


def _cred():
    return SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="tok"))


def test_ask_posts_four_inline_frames_to_the_deployment(monkeypatch, cfg):
    monkeypatch.setattr(qa, "agent_definition", lambda e: {"model": "qwen--qwen35-08b-2", "instructions": "AGENT INSTR"})
    monkeypatch.setattr(qa, "inline_frame", lambda u, **k: f"data:image/jpeg;base64,{u}")
    sess = Sess(resp={"choices": [{"message": {"content": "A bridge is visible in frame 2."}}]})
    out = qa.ask_qwen_agent(cfg, "any land bridges?", "5", "2", credential=_cred(), session=sess,
                            frame_urls=[f"f{i}" for i in range(10)])
    assert out == "A bridge is visible in frame 2."
    url, kw = sess.posts[0]
    assert url == "https://found-vision-1.services.ai.azure.com/openai/v1/chat/completions"
    assert kw["headers"]["Authorization"] == "Bearer tok"
    body = json.loads(kw["data"])
    assert body["model"] == "qwen--qwen35-08b-2" and body["messages"][0]["content"] == "AGENT INSTR"
    assert sum(p["type"] == "image_url" for p in body["messages"][1]["content"]) == 4


def test_ask_returns_none_without_frames_and_on_any_failure(monkeypatch, cfg):
    assert qa.ask_qwen_agent(cfg, "q", "5", "2", frame_urls=[]) is None
    monkeypatch.setattr(qa, "agent_definition", lambda e: {})
    monkeypatch.setattr(qa, "inline_frame", lambda u, **k: (_ for _ in ()).throw(RuntimeError("download failed")))
    assert qa.ask_qwen_agent(cfg, "q", "5", "2", credential=_cred(), session=Sess(),
                             frame_urls=["f0"]) is None


def test_agent_definition_falls_back_when_foundry_unreachable(monkeypatch):
    qa._definition_cache.clear()

    class Boom:
        def get_agent(self, name):
            raise RuntimeError("down")

    assert qa.agent_definition(ENDPOINT, client=Boom()) == {}
    qa._definition_cache.clear()


# --- integration with the chat synthesis ------------------------------------
def _agents(qwen_enabled=True):
    return FoundryAgents(SimpleNamespace(project_endpoint=ENDPOINT, qwen_enabled=qwen_enabled))


def test_run_qwen_agent_disabled_unconfigured_or_out_of_budget_returns_none(monkeypatch):
    import time

    called = []
    monkeypatch.setattr(qa, "ask_qwen_agent", lambda *a, **k: called.append(1) or "x")
    assert _agents(qwen_enabled=False).run_qwen_agent("q", "5") is None
    assert FoundryAgents(SimpleNamespace(project_endpoint=None, qwen_enabled=True)).run_qwen_agent("q", "5") is None
    assert _agents().run_qwen_agent("q", "5", deadline=time.monotonic() - 1) is None
    assert called == []
    assert _agents().run_qwen_agent("q", "5", "2") == "x"


def test_qwen_output_is_rolled_into_the_consolidated_synthesis(monkeypatch):
    agents = _agents()
    monkeypatch.setattr(type(agents), "configured", property(lambda self: False))
    for name, marker in (("run_connected_agent", "S"), ("run_function_tools", "F"), ("run_analyzer_tools", "A")):
        monkeypatch.setattr(agents, name, lambda *a, _m=marker, **k: _m)
    seen = {}
    monkeypatch.setattr(agents, "run_qwen_agent", lambda *a, **k: seen.update(args=a, kw=k) or "QWEN_MARKER")
    monkeypatch.setattr(agents, "_echo", lambda text: text)
    result = agents.synthesize_from_chat_agent("land bridges?", "5", video_id="2")
    assert "[Qwen Vision Agent Output]:\nQWEN_MARKER" in result
    assert seen["args"] == ("land bridges?", "5", "2") and "deadline" in seen["kw"]
