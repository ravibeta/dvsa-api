"""Azure AI Vision data-plane: image vectorization + analysis.

Ported from ``ezvision``'s ``videos/myvideoindexer.py`` (``vectorize_image`` and
``analyze_image``) into a cohesive :class:`VisionClient` driven by
:class:`~core.azure.config.AzureEnvironmentConfig`.

- :meth:`VisionClient.vectorize_image` calls the Computer Vision
  ``retrieval:vectorizeImage`` endpoint and pads to ``vector_dimensions`` (1536),
  exactly like the source. Offline (no vision key) it returns a deterministic
  pseudo-embedding so the RAG pipeline stays runnable.
- :meth:`VisionClient.analyze_image` calls ``ImageAnalysisClient`` for caption /
  tags / objects / dense captions; offline it derives a synthetic description.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .config import AzureEnvironmentConfig
from .offline import deterministic_embedding, retry

logger = logging.getLogger("apps.azure")


class VisionClient:
    """Vectorize and analyze aerial frames via Azure AI Vision (offline-safe)."""

    def __init__(self, config: AzureEnvironmentConfig) -> None:
        self.config = config

    @property
    def online(self) -> bool:
        c = self.config
        return bool(c.openai_endpoint is not None or self._vision_ready())

    def _vision_ready(self) -> bool:
        from django.conf import settings  # noqa: PLC0415

        return bool(
            getattr(settings, "AZURE_AI_VISION_ENDPOINT", None)
            and getattr(settings, "AZURE_AI_VISION_API_KEY", None)
        )

    def _vision_settings(self):
        from django.conf import settings  # noqa: PLC0415

        return (
            getattr(settings, "AZURE_AI_VISION_ENDPOINT", None),
            getattr(settings, "AZURE_AI_VISION_API_KEY", None),
            getattr(settings, "AZURE_AI_VISION_REGION", "eastus"),
        )

    # ----- vectorize -----------------------------------------------------
    @retry(reraise=True)
    def _vectorize_remote(self, image_url: str) -> Optional[List[float]]:
        import requests  # noqa: PLC0415

        endpoint, key, _region = self._vision_settings()
        url = (
            f"{endpoint}/computervision/retrieval:vectorizeImage"
            f"?api-version={self.config.vision_api_version}"
            f"&model-version={self.config.vision_model_version}"
        )
        headers = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": key}
        resp = requests.post(url, headers=headers, json={"url": image_url}, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("vector")
        logger.warning("vectorizeImage failed: %s %s", resp.status_code, resp.text[:200])
        raise RuntimeError(f"vectorizeImage error {resp.status_code}")

    def vectorize_image(self, image_url: str) -> List[float]:
        """Return a ``vector_dimensions``-wide embedding for ``image_url``.

        Pads short vectors with zeros to the index width (matching the source's
        ``np.pad(vector, (0, 1536 - len(vector)))``).
        """
        dims = self.config.vector_dimensions
        vector: Optional[List[float]] = None
        if self._vision_ready():
            try:
                vector = self._vectorize_remote(image_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vision vectorize fell back offline: %s", exc)
        if not vector:
            return deterministic_embedding(image_url, dims)
        if len(vector) < dims:
            vector = list(vector) + [0.0] * (dims - len(vector))
        return list(vector)[:dims]

    # ----- analyze -------------------------------------------------------
    @retry(reraise=True)
    def analyze_image(self, image_url: str) -> Dict[str, Any]:
        """Return ``{caption, tags, objects, dense_captions}`` for ``image_url``.

        Offline (no vision key) it returns a synthetic but well-formed result so
        downstream captioning/indexing keeps working.
        """
        if not self._vision_ready():
            tail = image_url.split("/")[-1].split("?")[0]
            return {
                "caption": f"aerial frame {tail}",
                "tags": ["aerial", "drone"],
                "objects": [],
                "dense_captions": [],
            }
        from azure.ai.vision.imageanalysis import ImageAnalysisClient  # noqa: PLC0415
        from azure.ai.vision.imageanalysis.models import VisualFeatures  # noqa: PLC0415
        from azure.core.credentials import AzureKeyCredential  # noqa: PLC0415

        endpoint, key, _region = self._vision_settings()
        client = ImageAnalysisClient(endpoint, AzureKeyCredential(key))
        result = client.analyze_from_url(
            image_url=image_url,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.READ,
                VisualFeatures.SMART_CROPS,
                VisualFeatures.PEOPLE,
            ],
            gender_neutral_caption=True,
        )
        caption = result.caption.text if result.caption else "No Caption"
        dense = (
            [c.text for c in result.dense_captions.list]
            if result.dense_captions else []
        )
        tags = [t.name for t in result.tags.list] if result.tags else []
        objects = (
            [o.tags[0].name for o in result.objects.list if o.tags]
            if result.objects else []
        )
        return {
            "caption": caption,
            "tags": tags,
            "objects": objects,
            "dense_captions": dense,
        }

    def analyze_image_description(self, image_url: str) -> str:
        """Compact JSON description string (source ``analyze_image`` shape)."""
        return json.dumps(self.analyze_image(image_url))
