"""Azure Video Indexer workflow (ported from ezvision videos/myvideoindexer.py).

A :class:`VideoIndexerClient` wraps the Video Indexer REST API: obtain an access
token, upload/index a video, poll insights, build a highlight project, render,
and download. :meth:`indexing_workflow` ties it together with the blob + vision
pipeline to populate the search index.

When the Video Indexer account/key is not configured, methods log and return
``None`` so the rest of the system degrades gracefully (offline-first).
"""

from __future__ import annotations

import logging
import random
import string
import time
import urllib.parse
from typing import List, Optional, Tuple

from .config import AzureEnvironmentConfig

logger = logging.getLogger("apps.azure")


class VideoIndexerClient:
    def __init__(self, config: AzureEnvironmentConfig) -> None:
        self.config = config

    @property
    def configured(self) -> bool:
        c = self.config
        return bool(c.video_indexer_account and (c.video_indexer_api_key or c.video_indexer_access_token))

    def _base(self) -> str:
        c = self.config
        return f"{c.video_indexer_url}/{c.video_indexer_region}/Accounts/{c.video_indexer_account}"

    # ----- auth ----------------------------------------------------------
    def get_access_token(self) -> Optional[str]:
        c = self.config
        if c.video_indexer_access_token:
            return c.video_indexer_access_token.strip('"')
        if not (c.video_indexer_account and c.video_indexer_api_key):
            return None
        import requests  # noqa: PLC0415

        url = f"{c.video_indexer_url}/auth/{c.video_indexer_region}/Accounts/{c.video_indexer_account}/AccessToken"
        resp = requests.get(url, headers={"Ocp-Apim-Subscription-Key": c.video_indexer_api_key}, timeout=60)
        return resp.text.strip('"')

    # ----- upload + insights --------------------------------------------
    def upload_and_index_video(self, access_token, video_file_path=None, video_url=None):
        import requests  # noqa: PLC0415

        video_name = None
        if video_url:
            video_name = urllib.parse.urlparse(video_url).path.split("/")[-1]
        elif video_file_path:
            video_name = _trim_filename(video_file_path.split("/")[-1])
        url = f"{self._base()}/Videos?name={video_name}&accessToken={access_token}"
        if video_url:
            encoded = urllib.parse.quote(video_url, safe="")
            url += f"&videoUrl={encoded}"
            headers = {
                "Ocp-apim-subscription-key": self.config.video_indexer_api_key or "",
                "Cache-Control": "no-cache",
                "Authorization": f"Bearer {access_token}",
            }
            return requests.post(url, headers=headers, timeout=120).json()
        with open(video_file_path, "rb") as fh:
            return requests.post(url, files={"file": fh}, timeout=600).json()

    def get_uploaded_video_id(self, access_token, video_file_path=None, video_url=None):
        data = self.upload_and_index_video(access_token, video_file_path, video_url)
        if "ErrorType" in data:
            logger.info("Video Indexer error: %s %s", data.get("ErrorType"), data.get("Message"))
        return data.get("id")

    def get_video_insights(self, access_token, video_id):
        import requests  # noqa: PLC0415

        url = f"{self._base()}/Videos/{video_id}/Index?accessToken={access_token}"
        count = 0
        while True:
            data = requests.get(url, timeout=60).json()
            if data.get("ErrorType") == "INVALID_VIDEO_ID":
                return None
            if data.get("state") == "Processed":
                return data
            count += 1
            if count % 10 == 0:
                logger.info("insights state=%s", data.get("state"))
            time.sleep(10)

    def reindex(self, access_token, video_id):
        import requests  # noqa: PLC0415

        url = f"{self._base()}/Videos/{video_id}/ReIndex?accessToken={access_token}"
        resp = requests.put(url, timeout=60)
        if resp.status_code == 200:
            return resp
        return self.get_video_insights(access_token, video_id)

    def get_selected_segments(self, insights, threshold) -> List[Tuple]:
        segments: List[Tuple] = []
        for video in insights["videos"]:
            for shot in video["insights"]["shots"]:
                for key_frame in shot["keyFrames"]:
                    inst = key_frame["instances"][0]
                    segments.append((inst["start"], inst["end"]))
        return segments

    def get_timestamps(self, access_token, video_id):
        insights = self.get_video_insights(access_token, video_id)
        return [
            (kf["instances"][0]["start"], kf["instances"][0]["end"])
            for kf in insights["videos"][0]["insights"]["shots"][0]["keyFrames"]
        ]

    # ----- projects / render --------------------------------------------
    def create_project(self, access_token, video_id, selected_segments):
        import requests  # noqa: PLC0415

        ranges = [{"videoId": video_id, "range": {"start": s, "end": e}}
                  for s, e in selected_segments]
        data = {
            "name": "".join(random.choices(string.hexdigits, k=8)),
            "videosRanges": ranges,
            "isSearchable": "false",
        }
        url = f"{self._base()}/Projects?accessToken={access_token}"
        resp = requests.post(url, json=data, headers={"Content-Type": "application/json"}, timeout=60)
        return resp.json().get("id") if resp.status_code == 200 else None

    def render_video(self, access_token, project_id):
        import requests  # noqa: PLC0415

        url = f"{self._base()}/Projects/{project_id}/render?sendCompletionEmail=false&accessToken={access_token}"
        resp = requests.post(url, headers={"Content-Type": "application/json"}, timeout=60)
        return resp if resp.status_code == 202 else None

    def get_render_operation(self, access_token, project_id):
        import requests  # noqa: PLC0415

        url = f"{self._base()}/Projects/{project_id}/renderoperation?accessToken={access_token}"
        while True:
            data = requests.get(url, timeout=60).json()
            if data.get("state") == "Succeeded":
                return data
            time.sleep(10)

    def download_rendered_file(self, access_token, project_id):
        import requests  # noqa: PLC0415

        url = f"{self._base()}/Projects/{project_id}/renderedfile/downloadurl?accessToken={access_token}"
        resp = requests.get(url, timeout=60)
        return resp.json().get("downloadUrl") if resp.status_code == 200 else None

    def index_and_download_video(self, account_id=None, project_id=None, video_id=None,
                                 video_file_path=None, video_url=None, repeat=True):
        access_token = self.get_access_token()
        if not access_token:
            logger.info("Video Indexer not configured; skipping index_and_download_video")
            return None
        if not video_id and not video_file_path and not video_url:
            return None
        if not video_id:
            video_id = self.get_uploaded_video_id(access_token, video_file_path, video_url)
        if not video_id:
            return None
        insights = self.reindex(access_token, video_id) if repeat else self.get_video_insights(access_token, video_id)
        segments = self.get_selected_segments(insights, 10)
        if not project_id:
            project_id = self.create_project(access_token, video_id, segments)
        if self.render_video(access_token, project_id):
            self.get_render_operation(access_token, project_id)
            return self.download_rendered_file(access_token, project_id)
        return None


def _trim_filename(filename: str, max_length: int = 255) -> str:
    import os  # noqa: PLC0415

    base, ext = os.path.splitext(filename)
    return base[: max_length - len(ext)] + ext
