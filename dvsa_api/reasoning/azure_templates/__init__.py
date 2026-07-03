"""Sample request/response templates + prompt wrappers for Foundry reasoning.

These are documentation-grade examples (loadable JSON) that show the expected
``context`` shape into :meth:`AzureFoundryAdapter.predict` and the structured
response it returns. They are also used by the test-suite fixtures.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

_HERE = os.path.dirname(__file__)


def load(name: str) -> Dict[str, Any]:
    """Load a bundled template JSON by base name (e.g. ``"infer_request"``)."""
    with open(os.path.join(_HERE, f"{name}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)
