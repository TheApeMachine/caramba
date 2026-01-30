"""Prompt-tuning adapter helpers for benchmarks.

Supports PEFT P-Tuning adapters that store `prompt_embeddings` in a
`.safetensors` file (adapter_model.safetensors).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor, nn

from console import logger
from infer.context import InferContext


def load_prompt_embeddings(path: Path) -> Tensor:
    """Load prompt embeddings from a PEFT adapter checkpoint."""
    data = safetensors_load_file(Path(path), device="cpu")
    if "prompt_embeddings" not in data:
        raise KeyError("PEFT adapter missing 'prompt_embeddings' key.")
    return data["prompt_embeddings"]


class PromptTuningAdapter(nn.Module):
    """Wrap a Caramba model with prompt-tuning virtual tokens."""

    def __init__(self, base_model: nn.Module, prompt_embeddings: Tensor) -> None:
        super().__init__()
        self.base_model = base_model
        # Register as buffer so device/dtype moves follow the module.
        self.register_buffer("prompt_embeddings", prompt_embeddings, persistent=False)

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_embeddings.shape[0])

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_input_embeddings(self) -> nn.Module | None:
        getter = getattr(self.base_model, "get_input_embeddings", None)
        return cast(nn.Module | None, getter()) if callable(getter) else None

    @property
    def vocab_size(self) -> int | None:
        return getattr(self.base_model, "vocab_size", None)

    def _resolve_prompt(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        prompt = self.prompt_embeddings.to(device=device, dtype=dtype)
        # (P, D) -> (B, P, D)
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x: Tensor, *, ctx: object | None = None, return_features: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if not hasattr(self.base_model, "embedder") or not hasattr(self.base_model, "topology"):
            raise RuntimeError("PromptTuningAdapter expects a Caramba Model-like base.")

        embedder = getattr(self.base_model, "embedder")
        topology = getattr(self.base_model, "topology")
        feats_to_logits = getattr(self.base_model, "_features_to_logits", None)
        if not callable(feats_to_logits):
            raise RuntimeError("Base model missing _features_to_logits; cannot adapt prompt-tuning.")

        # Compute token embeddings
        token_emb = embedder(x)
        if token_emb.ndim != 3:
            raise ValueError(f"Expected token embeddings with shape (B, T, D), got {token_emb.shape}")

        # In cache decode steps (pos_offset > 0), the prompt is already in KV-cache.
        # Only prepend virtual tokens for prefill (pos_offset == 0) or non-cache paths.
        if isinstance(ctx, InferContext) and int(ctx.pos_offset) > 0:
            x_emb = token_emb
        else:
            batch_size = int(token_emb.shape[0])
            prompt = self._resolve_prompt(
                batch_size=batch_size, device=token_emb.device, dtype=token_emb.dtype
            )
            x_emb = torch.cat([prompt, token_emb], dim=1)

        features = topology(x_emb, ctx=ctx)  # type: ignore[call-arg]
        logits = cast(Tensor, feats_to_logits(features))

        # Drop prompt logits so downstream losses align to input tokens.
        if logits.size(1) >= token_emb.size(1) + self.prompt_len:
            logits = logits[:, self.prompt_len :, :]

        if return_features:
            return features, logits
        return logits

    def __getattr__(self, name: str) -> Any:
        # First, let nn.Module resolve parameters/buffers/submodules.
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate missing attributes to the base model when present.
            try:
                base = super().__getattribute__("base_model")
            except AttributeError:
                raise
            if hasattr(base, name):
                return getattr(base, name)
            raise
