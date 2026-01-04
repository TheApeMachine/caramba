"""RandomModel provider that selects one of three models per request.

This ensures each persona's behavior is consistent (same instructions/tools)
while the underlying provider is randomized to reduce model bias.
"""
from __future__ import annotations

import random
from typing import Any

from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm


class RandomModel:
    """Model wrapper that randomly selects one of three providers per request.

    On each generation start, picks one model from the pool and uses it for
    the entire streaming response (no mid-stream switching).

    Supported model IDs:
    - openai/gpt-5.2
    - anthropic/claude-opus-4-5-20251101
    - google/gemini-3-pro-preview
    """

    # The three model IDs that RandomModel chooses from (per request).
    MODEL_POOL = [
        "openai/gpt-5.2",
        "anthropic/claude-opus-4-5-20251101",
        "google/gemini-3-pro-preview",
    ]

    def __init__(self) -> None:
        """Initialize RandomModel.

        The actual model instance is created lazily on first use (per request).
        """
        self._selected_model: Any | None = None
        self._selected_model_id: str | None = None

    def _select_model(self) -> Any:
        """Select a random model from the pool (once per request)."""
        if self._selected_model is None:
            model_id = random.choice(self.MODEL_POOL)
            self._selected_model_id = model_id

            # Normalize model ID and create appropriate ADK model instance.
            if model_id.startswith(("gemini/", "google/")):
                # ADK native Gemini integration expects bare model name.
                gemini_model = model_id.split("/", 1)[1].strip()
                self._selected_model = Gemini(model=gemini_model)
            else:
                # Route everything else (OpenAI, Anthropic) through LiteLLM.
                self._selected_model = LiteLlm(model=model_id)

        return self._selected_model

    def reset(self) -> None:
        """Reset the selected model (call before each new request)."""
        self._selected_model = None
        self._selected_model_id = None

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the selected model."""
        model = self._select_model()
        return getattr(model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate call to the selected model."""
        model = self._select_model()
        return model(*args, **kwargs)
