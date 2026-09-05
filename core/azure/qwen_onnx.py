"""Local ONNX backend for the Qwen tool — standalone, Azure-free inference.

When ``DVSA_QWEN_BACKEND=onnx`` the :func:`core.azure.analyzer.ask_qwen_vlm`
tool routes here instead of the Azure AI Foundry endpoint, letting dvsa-api serve
Qwen answers on a local deployment with no cloud dependency. The model
(``Qwen3.5-0.8B`` exported to ONNX) is loaded from ``DVSA_QWEN_ONNX_MODEL_PATH``
via `onnxruntime-genai <https://github.com/microsoft/onnxruntime-genai>`_.

The whole ``onnxruntime-genai`` interaction lives inside :func:`_default_generator`,
and :func:`run_local_qwen` accepts a ``generator_factory`` override, so tests (and
CI) exercise the routing/fallback logic fully offline — no ``.onnx`` weights and no
``onnxruntime-genai`` install required — mirroring the ``session_factory`` seam used
by ``custom_models.adapters.onnx_adapter``.

Like the Azure path, any missing configuration or runtime error degrades to a
benign ``"No comment."`` so the agent pipeline stays runnable.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger("apps.azure")

SYSTEM_PROMPT = "You are an aerial drone image and vision analyst."

# Recommended Qwen3.5 generation settings (match the Azure chat path).
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_LENGTH = 1024

# ``(model_path, system_prompt, user_prompt) -> assistant_text``.
GeneratorFactory = Callable[[str, str, str], str]


def _default_generator(model_path: str, system_prompt: str, user_prompt: str) -> str:
    """Run one chat turn through a local Qwen3.5 ONNX model.

    Lazily imports ``onnxruntime_genai`` (an optional, opt-in dependency) so the
    module stays importable without it. Uses the Qwen ChatML template and the
    recommended generation settings, and returns only the assistant's reply.
    """
    import onnxruntime_genai as og  # noqa: PLC0415

    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P
    )

    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    new_tokens = []
    while not generator.is_done():
        generator.generate_next_token()
        new_tokens.append(generator.get_next_tokens()[0])

    return tokenizer.decode(new_tokens).strip()


def run_local_qwen(
    model_path: str,
    query_text: str,
    *,
    generator_factory: Optional[GeneratorFactory] = None,
) -> str:
    """Answer ``query_text`` with a locally hosted Qwen3.5-0.8B ONNX model.

    Returns the model's reply, or ``"No comment."`` when the model path is unset,
    ``onnxruntime-genai`` is not installed, or generation fails — keeping the tool
    safe to call on a machine that has not (yet) provisioned the local model.
    """
    if not model_path:
        logger.info("Qwen onnx backend selected but no model path configured; no comment")
        return "No comment."

    factory = generator_factory or _default_generator
    try:
        text = factory(model_path, SYSTEM_PROMPT, query_text)
    except Exception as exc:  # noqa: BLE001
        logger.info("Local Qwen ONNX inference failed: %s", exc)
        return "No comment."
    return text or "No comment."
