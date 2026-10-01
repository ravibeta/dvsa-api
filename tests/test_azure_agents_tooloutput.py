"""Unit tests for FoundryAgents tool-output coercion.

Foundry's ``submit_tool_outputs`` JSON-encodes every ToolOutput, which throws
``TypeError: Object of type ndarray is not JSON serializable`` when an analyzer
function returns a NumPy array. ``_coerce_tool_output`` normalizes any tool
return value into a JSON-safe string so the agent run can continue.
"""

import json

import numpy as np

from core.azure.agents import _coerce_tool_output


def test_ndarray_becomes_json_list():
    out = _coerce_tool_output(np.array([1, 2, 3]))
    assert isinstance(out, str)
    assert json.loads(out) == [1, 2, 3]


def test_nested_ndarray_in_dict_serializes():
    out = _coerce_tool_output({"matches": np.array([[1, 2], [3, 4]]), "count": 2})
    assert isinstance(out, str)
    assert json.loads(out) == {"matches": [[1, 2], [3, 4]], "count": 2}


def test_numpy_scalar_becomes_native():
    out = _coerce_tool_output(np.int64(7))
    assert json.loads(out) == 7


def test_string_passes_through_unchanged():
    assert _coerce_tool_output("already a string") == "already a string"


def test_none_passes_through():
    assert _coerce_tool_output(None) is None


def test_unserializable_falls_back_to_str():
    class Weird:
        def __repr__(self):
            return "<weird>"

    # No tolist/item and not JSON-native -> str() fallback via _json_safe.
    out = _coerce_tool_output(Weird())
    assert isinstance(out, str)
    assert "weird" in out
