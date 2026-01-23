"""Language model system.

This wraps the existing `model.Model` into a generic "system" shape:
- accepts dict-like batches
- returns dict-like outputs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from console import logger

from compiler.lower import Lowerer
from compiler.validate import Validator
from config.model import ModelConfig
from model import Model
from initializers.registry import InitializerRegistry


@dataclass
class LanguageModelSystem:
    """A system that produces next-token logits from token batches.

    Config:
    - model: ModelConfig payload (required)
    """

    model: dict[str, Any]
    weight_init: dict[str, Any] | None = None
    return_features: bool = False
    features_key: str = "features"

    def __post_init__(self) -> None:
        cfg = ModelConfig.model_validate(self.model)
        # weight_init from constructor kwargs is likely a raw dict or None.
        # But ModelConfig also has weight_init.
        # To avoid confusion, we rely on ModelConfig having the correct weight_init config
        # because the registry/builder logic likely populated the 'model' dict or passed it separately.
        #
        # Actually, ModelConfig structure puts 'weight_init' adjacent to 'model' (topology etc) in the
        # higher-level system config? No, looking at `config/model.py`, `ModelConfig` HAS `weight_init`.
        #
        # Wait, the `LanguageModelSystem` receives args from `ExperimentTargetConfig.system.config`.
        # If `ExperimentTargetConfig.system.config` has `weight_init` separate from `model`, then
        # `LanguageModelSystem` needs to accept it.
        #
        # If `ModelConfig` encapsulates everything, then `LanguageModelSystem` should just take
        # `**config` and parse it into `ModelConfig`.
        #
        # The current implementation of `LanguageModelSystem` takes `model: dict`.
        # The error says "Config keys: ['model', 'weight_init']".
        # This implies `system.config` in the manifest has:
        #   model: {...}
        #   weight_init: {...}
        #
        # But `LanguageModelSystem` expects `model` to be the ModelConfig payload?
        #
        # Let's fix `LanguageModelSystem` to accept optional `weight_init` and merge it into the config if needed.

        # If weight_init was passed explicitly, we should probably update the parsed config.
        if self.weight_init is not None:
            # We need to make sure this merges correctly if we are validating `self.model`.
            # But wait, `self.model` is `dict[str, Any]`.
            # If `weight_init` is a top-level key in `system.config`, we can just inject it into
            # the dict before validation if `ModelConfig` expects it.
            #
            # However, `LanguageModelSystem` does:
            # `cfg = ModelConfig.model_validate(self.model)`
            #
            # So `self.model` is expected to be the FULL ModelConfig payload.
            # BUT the manifest (herorun.yml) has:
            # system:
            #   config:
            #     weight_init: ...
            #     model: ...
            #
            # The Registry unflatpacks `system.config` into kwargs.
            # So we get `model={...}` and `weight_init={...}`.
            #
            # But `ModelConfig.model_validate(self.model)` implies `self.model` should contain `weight_init` too
            # if `ModelConfig` has that field.
            #
            # We have two choices:
            # 1. Update manifest to put `weight_init` INSIDE `model`.
            # 2. Update `LanguageModelSystem` to accept `weight_init` and inject it into the config.
            #
            # Option 2 is safer because `herorun.yml` structure was:
            # config:
            #   weight_init: ...
            #   model: ...

            if "weight_init" not in self.model:
                self.model["weight_init"] = self.weight_init

        cfg = ModelConfig.model_validate(self.model)
        cfg = Lowerer().lower_model(cfg)
        Validator().validate_model_config(cfg)
        self.config = cfg
        self.module: nn.Module = Model(cfg)
        self._apply_initialization()

    def _apply_initialization(self) -> None:
        """Apply weight initialization based on configuration."""
        initializer = InitializerRegistry.build(self.config.weight_init)
        initializer.initialize(self.module)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "LanguageModelSystem":
        self.module = self.module.to(device=device, dtype=dtype)
        return self

    def train(self, mode: bool = True) -> None:
        self.module.train(mode)

    def eval(self) -> None:
        self.module.eval()

    def forward(self, batch: dict[str, Any], *, ctx: object | None = None) -> dict[str, Any]:
        input_ids: Tensor = batch["input_ids"]
        # Only request features when a consumer exists (diffusion head training).
        # Doing so unconditionally changes output structure and can hurt torch.compile stability.
        diffusion_head = getattr(self.config, "diffusion_head", None)
        want_features = bool(getattr(diffusion_head, "enabled", False) or self.return_features)
        if want_features:
            try:
                if hasattr(self.module, "forward_with_hidden"):
                    features, logits = self.module.forward_with_hidden(input_ids, ctx=ctx)  # type: ignore[call-arg]
                else:
                    result = self.module(input_ids, ctx=ctx, return_features=True)  # type: ignore[call-arg]
                    if not (isinstance(result, tuple) and len(result) == 2):
                        raise TypeError("Expected Model(return_features=True) to return (features, out)")
                    features, logits = result
                out = {str(self.features_key): features, "logits": logits, "_system": self.module}
                # Attach MOSAIC aux outputs when present on ctx.
                aux = getattr(ctx, "memblock_aux_out", None) if ctx is not None else None
                if isinstance(aux, dict):
                    for k, v in aux.items():
                        if isinstance(k, str) and isinstance(v, Tensor):
                            out[k] = v
                return out
            except TypeError as e:
                # Don't hide real signature/dispatch issues; this is a common source of
                # silent feature disablement (e.g., return_features not supported).
                logger.warning(f"Model(return_features=True) TypeError; falling back: {e!r}")
        logits = self.module(input_ids, ctx=ctx)  # type: ignore[call-arg]
        out2 = {"logits": logits, "_system": self.module}
        aux2 = getattr(ctx, "memblock_aux_out", None) if ctx is not None else None
        if isinstance(aux2, dict):
            for k, v in aux2.items():
                if isinstance(k, str) and isinstance(v, Tensor):
                    out2[k] = v
        return out2

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        """Return named parameters for gradient isolation trainer."""
        return self.module.named_parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)

