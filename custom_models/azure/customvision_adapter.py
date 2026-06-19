"""Azure Custom Vision adapter for the ``custom_models`` package.

Custom Vision projects are trained in the portal/SDK and **exported** as ONNX;
you cannot upload an arbitrary ONNX into an existing project. This adapter:

* re-uses the secret-free export helper from
  :mod:`custom_model.azure_customvision_helper` (no keys are hard-coded — pass
  them in from a secret store), and
* turns a downloaded ONNX artifact into a :class:`~custom_models.loader.ModelSpec`
  that the registry's :func:`~custom_models.registry.get_detector` can run via
  the ONNX adapter.

Typical flow::

    adapter = CustomVisionAdapter(project_id, iteration_id, training_key, endpoint)
    uri = adapter.export_onnx()                 # async export -> download URI
    # ... download + unzip the package to model.onnx + labels.txt ...
    spec = spec_from_export("model.onnx", labels_file="label_map.json")
    detector = get_detector(spec).load()
"""

from __future__ import annotations

import logging
from typing import Optional

# Re-use the existing, tested export helper rather than duplicating the REST calls.
from custom_model.azure_customvision_helper import export_iteration_to_onnx

from ..loader import ModelSpec

logger = logging.getLogger(__name__)

__all__ = ["CustomVisionAdapter", "export_iteration_to_onnx", "spec_from_export"]


def spec_from_export(
    onnx_path: str,
    *,
    labels_file: Optional[str] = None,
    model_id: str = "azure-customvision",
    name: str = "Azure Custom Vision (ONNX export)",
    input_size=(320, 320),
    capabilities=None,
) -> ModelSpec:
    """Build a :class:`ModelSpec` for a downloaded Custom Vision ONNX export.

    Custom Vision compact/general object-detection exports typically use a
    ``320x320`` input — adjust ``input_size`` if your domain differs.
    """

    return ModelSpec(
        id=model_id,
        name=name,
        format="onnx",
        path=onnx_path,
        labels_file=labels_file,
        input_size=input_size,
        capabilities=list(capabilities or []),
        task="detection",
        source_url="https://learn.microsoft.com/azure/ai-services/custom-vision-service/",
    )


class CustomVisionAdapter:
    """Thin wrapper to export a trained Custom Vision iteration as ONNX.

    Parameters
    ----------
    project_id, iteration_id:
        Identify the trained model to export.
    training_key:
        Custom Vision **training** key (used for the export call).
    endpoint:
        Resource endpoint, e.g. ``https://<region>.api.cognitive.microsoft.com``.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        project_id: str,
        iteration_id: str,
        training_key: str,
        endpoint: str,
        *,
        timeout: int = 30,
    ) -> None:
        self.project_id = project_id
        self.iteration_id = iteration_id
        self.training_key = training_key
        self.endpoint = endpoint
        self.timeout = timeout

    def export_onnx(self) -> str:
        """Trigger an ONNX export and return its ``downloadUri`` once ready."""

        logger.info("Exporting Custom Vision iteration %s as ONNX", self.iteration_id)
        return export_iteration_to_onnx(
            project_id=self.project_id,
            iteration_id=self.iteration_id,
            training_key=self.training_key,
            endpoint=self.endpoint,
            timeout=self.timeout,
        )

    def spec_from_export(self, onnx_path: str, **kwargs) -> ModelSpec:
        """Convenience: build a :class:`ModelSpec` for an already-downloaded export."""

        return spec_from_export(onnx_path, **kwargs)
