"""Unit tests for ``describe_frame`` — the real captioning function tool.

The chat agent previously had no way to answer "what's in the scene" /
"how many X are visible" questions: its only tools were object-counting (ORB
template matching, needs a user-supplied object image) or third-party
multimodal APIs (Perplexity/Qwen) that may not be configured. ``describe_frame``
resolves the real frame SAS URL for account_id/video_id and reuses
``VisionClient.analyze_image`` — the same Azure AI Vision resource already used
to caption/vectorize frames during ingest — so no extra credentials are needed.
"""

from unittest.mock import MagicMock

import core.azure.analyzer as analyzer_module
import core.azure.blob as blob_module
import core.azure.vision as vision_module
from core.azure.analyzer import analyzer_functions, describe_frame


def test_describe_frame_resolves_url_and_calls_vision(monkeypatch):
    monkeypatch.setattr(analyzer_module, "_cfg", lambda: "dummy-config")
    monkeypatch.setattr(
        analyzer_module, "get_sas_url_template",
        lambda account_id, video_id=None: "https://x/input/5/images/2/frame(number).jpg?sig=abc",
    )
    monkeypatch.setattr(
        blob_module, "get_sas_url_for_frame",
        lambda template, frame_number: template.replace("frame(number)", f"frame{frame_number}"),
    )
    fake_client = MagicMock()
    fake_client.analyze_image_description.return_value = (
        '{"caption": "a railway track", "tags": ["rail"], "objects": [], "dense_captions": []}'
    )
    fake_vision_client_cls = MagicMock(return_value=fake_client)
    monkeypatch.setattr(vision_module, "VisionClient", fake_vision_client_cls)

    result = describe_frame("5", video_id="2", frame_number=0)

    fake_vision_client_cls.assert_called_once_with("dummy-config")
    fake_client.analyze_image_description.assert_called_once_with(
        "https://x/input/5/images/2/frame0.jpg?sig=abc"
    )
    assert "railway track" in result


def test_describe_frame_without_frames_returns_message(monkeypatch):
    monkeypatch.setattr(analyzer_module, "_cfg", lambda: "dummy-config")
    monkeypatch.setattr(
        analyzer_module, "get_sas_url_template", lambda account_id, video_id=None: None,
    )

    result = describe_frame("5", video_id="999")

    assert "No extracted frames" in result


def test_describe_frame_is_registered_as_a_tool():
    assert describe_frame in analyzer_functions()
