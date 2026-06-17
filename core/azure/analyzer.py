"""CV analyzer functions + Perplexity retrieval (ported from ezvision).

Ports ``videos/analyzer_functions.py``. The classical CV pieces (template
matching, ORB feature matching, density clustering) reuse the project's existing
``apps.analytics.routines`` where they overlap, and otherwise rely on OpenCV +
scikit-learn (which already ships HDBSCAN, so no extra dependency).

The function-tool entry points (:func:`agentic_retrieval`, :func:`ask_perplexity`,
:func:`get_object_uri`, :func:`get_scene_uri`) keep their original signatures so
they can be registered directly as Foundry ``FunctionTool`` callables. They read
Azure config from Django settings internally (via ``AzureEnvironmentConfig``).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional, Set

from .config import AzureEnvironmentConfig

logger = logging.getLogger("apps.azure")

MATCH_THRESHOLD = 0.65
MIN_CLUSTER_MEMBERS = 2


def _cfg() -> AzureEnvironmentConfig:
    return AzureEnvironmentConfig.from_settings()


# --------------------------------------------------------------------------
# Image download + template matching
# --------------------------------------------------------------------------
def download_image(url):
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import requests  # noqa: PLC0415

    resp = requests.get(url, timeout=60)
    arr = np.frombuffer(resp.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


load_image_from_sas = download_image


def count_object_occurrences(scene, template, threshold=MATCH_THRESHOLD):
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    w, h = template_gray.shape[::-1]
    rects = [[pt[0], pt[1], pt[0] + w, pt[1] + h] for pt in zip(*locations[::-1])]
    rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)
    return len(rects)


def count_matches(scene_uri, object_uri):
    return count_object_occurrences(download_image(scene_uri), download_image(object_uri))


# --------------------------------------------------------------------------
# ORB feature matching + clustering (reuses analytics routines for clustering)
# --------------------------------------------------------------------------
def _keypoints_and_descriptors(scene_img, object_img):
    import cv2  # noqa: PLC0415

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(object_img, None)
    kp2, des2 = orb.detectAndCompute(scene_img, None)
    if des1 is None or des2 is None:
        return None, None, None, None
    return kp1, des1, kp2, des2


def get_matched_descriptors(scene_img, object_img):
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    kp1, des1, kp2, des2 = _keypoints_and_descriptors(scene_img, object_img)
    if des1 is None or des2 is None:
        return np.array([])
    matches = sorted(
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2),
        key=lambda x: x.distance,
    )
    return np.array([des2[m.trainIdx] for m in matches])


def cluster_by_similarity(descriptors, min_cluster_size=MIN_CLUSTER_MEMBERS):
    """Cluster ORB descriptors by cosine similarity using scikit-learn HDBSCAN."""
    import numpy as np  # noqa: PLC0415

    if len(descriptors) == 0:
        return np.array([])
    from sklearn.preprocessing import normalize  # noqa: PLC0415

    descriptors = normalize(descriptors, norm="l2")
    try:
        from sklearn.cluster import HDBSCAN  # noqa: PLC0415

        return HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(descriptors)
    except Exception:  # noqa: BLE001 - older sklearn: fall back to DBSCAN
        from sklearn.cluster import DBSCAN  # noqa: PLC0415

        return DBSCAN(eps=0.5, min_samples=min_cluster_size).fit_predict(descriptors)


def count_multiple_matches(scene_uri, object_uri):
    descriptors = get_matched_descriptors(load_image_from_sas(scene_uri), load_image_from_sas(object_uri))
    labels = cluster_by_similarity(descriptors)
    return len([1 for label in labels if label == 1])


def agentic_retrieval(pattern_uri: Optional[str] = None, content_uri: Optional[str] = None,
                      query_text: Optional[str] = None, account_id: Optional[str] = None,
                      video_id: Optional[str] = None) -> str:
    """Count occurrences of an object pattern within a scene (function tool)."""
    if not pattern_uri:
        pattern_uri = get_object_uri(query_text, account_id, video_id)
    if not content_uri:
        content_uri = get_scene_uri(query_text, account_id, video_id)
    if not (pattern_uri and content_uri):
        return "0"
    return f"{count_multiple_matches(content_uri, pattern_uri)}"


# --------------------------------------------------------------------------
# Bounding-box parsing
# --------------------------------------------------------------------------
def parse_bbox(s: str):
    for pattern in (
        r"\{x:\s*(\d+),\s*y:\s*(\d+),\s*w:\s*(\d+),\s*h:\s*(\d+)\}",
        r"\(x:\s*(\d+),\s*y:\s*(\d+),\s*width:\s*(\d+),\s*height:\s*(\d+)\)",
    ):
        m = re.search(pattern, s)
        if m:
            return tuple(map(int, m.groups()))
    return None


# --------------------------------------------------------------------------
# Perplexity multimodal retrieval
# --------------------------------------------------------------------------
def perplexity_retrieval(images_uri, query_text, account_number=2, frames=None,
                         image_uri_template=None, pattern="(number)"):
    import requests  # noqa: PLC0415

    cfg = _cfg()
    if not cfg.perplexity_chat_api_key:
        logger.info("Perplexity not configured; returning no comment")
        return "No comment."
    headers = {
        "Authorization": f"Bearer {cfg.perplexity_chat_api_key}",
        "accept": "application/json", "content-type": "application/json",
    }
    content = [{"type": "text", "text": query_text}]
    for i, uri in enumerate(images_uri or []):
        if i >= 20:
            break
        content.append({"type": "image_url", "image_url": uri})
    if frames and image_uri_template:
        for i, fr in enumerate(frames):
            if i >= 20:
                break
            content.append({"type": "image_url",
                            "image_url": image_uri_template.replace(pattern, fr)})
    payload = {"model": "sonar-pro", "return_images": "true",
               "messages": [{"role": "user", "content": content}], "stream": False}
    try:
        resp = requests.post(cfg.perplexity_chat_api_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        logger.info("Perplexity request failed: %s", exc)
        return "No comment."


def ask_perplexity(query_text, account_id="2", video_id=None, frames_list=None):
    """Resolve frame SAS URLs for a video and query Perplexity over them."""
    cfg = _cfg()
    from .blob import container_sas_url, get_image_blob_url, get_uploaded_frames  # noqa: PLC0415

    try:
        from apps.videos.models import VideoEntity  # noqa: PLC0415

        if not video_id:
            video_id = str(VideoEntity.objects.filter(account_id=account_id).last().id)
        video_sas_url = VideoEntity.objects.get(pk=video_id).sas_url
    except Exception as exc:  # noqa: BLE001
        logger.info("ask_perplexity could not resolve video: %s", exc)
        return None
    video_sas_url = container_sas_url(cfg, video_sas_url)
    template = get_image_blob_url(video_sas_url, 0, folder="images", prefix="frame",
                                  include_name=False, video_id=video_id)
    highest = get_uploaded_frames(cfg, video_sas_url, account_id=account_id, video_id=video_id)
    frames: List[str] = []
    if highest and int(highest) > 0:
        frames = [str(0), str(int(highest / 2)), str(highest - 1)]
    if frames_list:
        frames = frames_list.strip(",").split(",")
    if not frames:
        frames = [str(n) for n in range(20)]
    template = template.replace("frame0", "frame(number)")
    return perplexity_retrieval(None, query_text, account_id, frames, template, pattern="(number)")


# --------------------------------------------------------------------------
# Object / scene URI resolution (agent-assisted)
# --------------------------------------------------------------------------
def get_sas_url_template(account_id, video_id=None, upload=False):
    cfg = _cfg()
    from .blob import container_sas_url, get_image_blob_url, get_uploaded_frames  # noqa: PLC0415

    try:
        from apps.videos.models import VideoEntity  # noqa: PLC0415

        if not video_id:
            video_id = str(VideoEntity.objects.filter(account_id=account_id).last().id)
        video_sas_url = VideoEntity.objects.get(pk=video_id).sas_url
        video_sas_url = container_sas_url(cfg, video_sas_url, upload=upload)
        template = get_image_blob_url(video_sas_url, 0, folder="images", prefix="frame",
                                      include_name=False, video_id=video_id)
        return template.replace("frame0", "frame(number)")
    except Exception as exc:  # noqa: BLE001
        logger.info("get_sas_url_template failed: %s", exc)
        return None


def get_object_uri(object_description, account_id, video_id=None, frame_number=None):
    """Find an object's bounding box via the scene-search agent, clip + upload it."""
    from .blob import get_sas_url_for_frame  # noqa: PLC0415

    sas_url_template = get_sas_url_template(account_id, video_id, upload=True)
    if frame_number and sas_url_template:
        return get_sas_url_for_frame(sas_url_template, frame_number)
    cfg = _cfg()
    from .agents import FoundryAgents  # noqa: PLC0415

    query_text = (f"Find {object_description} in saved images, cite your reference and "
                  f"from its description find the bounding box. Display it as {{x: , y: , w: , h: }}.")
    messages = FoundryAgents(cfg).ask_agent("scene-search-agent", query_text)
    if not messages:
        return None
    answer = url = None
    for message in messages:
        if message.text_messages:
            answer = message.text_messages[-1].text.value
            for annotation in message.text_messages[-1].text.annotations:
                if annotation.type == "url_citation":
                    url = annotation.url_citation.url
                    break
    if not (answer and url):
        return None
    bbox = parse_bbox(answer)
    if not bbox:
        return None
    return _clip_and_upload_object(cfg, account_id, url, bbox, sas_url_template, video_id)


def _clip_and_upload_object(cfg, account_id, url, bbox, sas_url_template, video_id):
    import cv2  # noqa: PLC0415
    from datetime import datetime  # noqa: PLC0415

    from .blob import (get_image_blob_url, get_sas_url_for_frame,  # noqa: PLC0415
                       read_image_from_blob, upload_image_to_blob)

    frame_number = str(int(url.split("-")[1]))
    image_url = get_sas_url_for_frame(sas_url_template, frame_number)
    image = read_image_from_blob(image_url)
    if image is None or not image.any():
        return None
    x, y, w, h = bbox
    clipped = image[y:y + h, x:x + w]
    if not clipped.any():
        return None
    _, buffer = cv2.imencode(".jpg", clipped)
    destination_file = datetime.now().strftime("%Y%m%d%H%M%S")
    object_url = get_image_blob_url(image_url, frame_number, folder="queries",
                                    prefix=destination_file, include_name=False)
    upload_image_to_blob(buffer.tobytes(), object_url)
    return object_url


def get_scene_uri(query_text, account_id, video_id=None, frame_number=None):
    from .blob import get_sas_url_for_frame  # noqa: PLC0415

    sas_url_template = get_sas_url_template(account_id, video_id)
    if frame_number and sas_url_template:
        return get_sas_url_for_frame(sas_url_template, frame_number)
    cfg = _cfg()
    from .agents import FoundryAgents  # noqa: PLC0415

    document = FoundryAgents(cfg).ask_agent_for_url(
        "scene-search-agent",
        f"Find a saved image for {query_text} and cite the document url where it was found",
    )
    if document and sas_url_template:
        frame_number = str(int(document.split("-")[1]))
        return get_sas_url_for_frame(sas_url_template, frame_number)
    return None


# --------------------------------------------------------------------------
# Foundry FunctionTool registries (callables exposed to agents)
# --------------------------------------------------------------------------
def analyzer_functions() -> Set[Callable[..., Any]]:
    return {
        download_image, count_object_occurrences, count_matches,
        get_matched_descriptors, cluster_by_similarity, count_multiple_matches,
        agentic_retrieval, get_object_uri, get_scene_uri, get_sas_url_template,
        ask_perplexity,
    }


def image_user_functions() -> Set[Callable[..., Any]]:
    return {agentic_retrieval, ask_perplexity}
