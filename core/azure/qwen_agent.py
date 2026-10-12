"""Run ``qwen-agent-in-a-team`` — the Foundry vision agent on the managed Qwen deployment.

Why this isn't a normal agent run: the project's managed-compute Qwen deployment
(``qwen--qwen35-08b-2``) accepts images through the OpenAI ``chat/completions``
protocol, but Foundry's Agents/Responses layers reject image content for it
("does not support image inputs", the deployment isn't flagged vision-capable).
So the agent is registered in Foundry as the definition of record
(``solution-accelerator/agents/qwen-agent-in-a-team.yaml`` — instructions and
model), and at query time this module sends *those* instructions plus the
selected frames **inline** (base64 data URLs; the model can't fetch SAS URLs)
straight to the deployment.

Only a few frames are sent: the curated extract-frames set is down-selected to
:data:`MAX_FRAMES` evenly spaced frames, each downscaled to ``MAX_SIDE`` px JPEG,
which keeps the request small enough for a 0.8B model.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any, List, Optional, Sequence

logger = logging.getLogger("apps.azure")

AGENT_NAME = "qwen-agent-in-a-team"
DEFAULT_MODEL = "qwen--qwen35-08b-2"
DEFAULT_INSTRUCTIONS = (
    "You are an aerial drone image analyst. The user message contains a question "
    "followed by a small set of video frames attached inline as images, in time "
    "order. Answer the question from what is visibly present."
)
MAX_FRAMES = 4
MAX_SIDE = 512
JPEG_QUALITY = 70
MAX_TOKENS = 600
REQUEST_TIMEOUT = 25
_DEFINITION_TTL = 300.0

_definition_cache: dict = {}


def select_frames(urls: Sequence[str], limit: int = MAX_FRAMES) -> List[str]:
    """Pick up to ``limit`` evenly spaced items, always keeping first and last."""
    urls = list(urls)
    if limit <= 0 or not urls:
        return []
    if len(urls) <= limit:
        return urls
    if limit == 1:
        return [urls[0]]
    step = (len(urls) - 1) / (limit - 1)
    return [urls[round(i * step)] for i in range(limit)]


def inline_frame(url: str, *, max_side: int = MAX_SIDE, quality: int = JPEG_QUALITY,
                 session: Any = None) -> str:
    """Download a frame and return it as a downscaled ``data:image/jpeg;base64,`` URL."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import requests  # noqa: PLC0415

    resp = (session or requests).get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"not a decodable image: {url.split('?')[0]}")
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _account_origin(project_endpoint: str) -> str:
    m = re.match(r"(https://[^/]+)/api/projects/", project_endpoint or "")
    if not m:
        raise ValueError("AZURE_PROJECT_ENDPOINT is not a Foundry project endpoint")
    return m.group(1)


def agent_definition(project_endpoint: str, *, client: Any = None) -> dict:
    """The registered agent's latest definition (instructions + model), cached briefly."""
    cached = _definition_cache.get(project_endpoint)
    if cached and time.monotonic() - cached[0] < _DEFINITION_TTL:
        return cached[1]
    definition: dict = {}
    try:
        from .foundry_provisioning import FoundryProjectClient  # noqa: PLC0415

        agent = (client or FoundryProjectClient(project_endpoint, retries=0)).get_agent(AGENT_NAME)
        definition = (((agent or {}).get("versions") or {}).get("latest") or {}).get("definition") or {}
    except Exception as exc:  # noqa: BLE001 - fall back to the built-in defaults
        logger.info("could not read %s definition from Foundry: %s", AGENT_NAME, exc)
    _definition_cache[project_endpoint] = (time.monotonic(), definition)
    return definition


def build_messages(query_text: str, instructions: str, images: Sequence[str]) -> List[dict]:
    note = (f"{len(images)} frame(s) attached, in time order (frame 1 is the earliest)."
            if images else "No frames are available.")
    content: List[dict] = [{"type": "text", "text": f"{query_text}\n\n{note}"}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
    return [{"role": "system", "content": instructions},
            {"role": "user", "content": content}]


def parse_completion(body: dict) -> Optional[str]:
    """Answer text from a chat completion.

    The deployment is a reasoning model: when its thinking isn't separated, the
    answer arrives in ``reasoning`` with ``content`` null — accept either.
    """
    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    text = (msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
    return text or None


def ask_qwen_agent(config: Any, query_text: str, account_id: str,
                   video_id: Optional[str] = None, *, timeout: float = REQUEST_TIMEOUT,
                   credential: Any = None, session: Any = None,
                   frame_urls: Optional[Sequence[str]] = None) -> Optional[str]:
    """Ask the Qwen vision agent about the selected curated frames; ``None`` on any failure."""
    try:
        import requests  # noqa: PLC0415

        from .foundry_provisioning import AI_SCOPE  # noqa: PLC0415

        if frame_urls is None:
            from .analyzer import curated_frame_urls  # noqa: PLC0415

            frame_urls = curated_frame_urls(account_id, video_id)
        selected = select_frames(frame_urls)
        if not selected:
            logger.info("qwen agent: no extracted frames for account %s video %s", account_id, video_id)
            return None
        images = [inline_frame(u) for u in selected]

        endpoint = config.project_endpoint
        definition = agent_definition(endpoint)
        if credential is None:
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415

            credential = DefaultAzureCredential()
        body = {
            "model": definition.get("model") or DEFAULT_MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": build_messages(
                query_text, definition.get("instructions") or DEFAULT_INSTRUCTIONS, images),
        }
        resp = (session or requests).post(
            f"{_account_origin(endpoint)}/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {credential.get_token(AI_SCOPE).token}",
                     "Content-Type": "application/json"},
            data=json.dumps(body), timeout=timeout)
        resp.raise_for_status()
        return parse_completion(resp.json())
    except Exception as exc:  # noqa: BLE001 - this agent must never break the chat turn
        logger.warning("qwen agent failed: %s", exc)
        return None
