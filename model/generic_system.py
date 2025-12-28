"""Generic ModelConfig-driven system.

`system.generic` is the bridge between Caramba's compiler-friendly `ModelConfig`
and the Phase-1 TensorDict protocol:

- accepts `batch: dict[str, Any]` (expects a tensor at `input_key`)
- returns `outputs: dict[str, Any]` (writes a tensor at `output_key`)

Unlike `system.language_model`, this makes **no assumption** about token IDs or
the presence of `input_ids`; it simply forwards the chosen input tensor into the
configured model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from compiler.lower import Lowerer
from compiler.validate import Validator
from config.model import ModelConfig
from model import Model


@dataclass
class GenericSystem:
    """A generic system backed by `model.Model` (ModelConfig).

    Config:
    - model: ModelConfig payload (required)
    - input_key: batch key holding the model input tensor (default: "inputs")
    - output_key: outputs key to store the primary tensor (default: "logits")
    - return_features: if True, also return intermediate features (default: False)
    - features_key: outputs key for features when return_features=True (default: "features")
    - include_system: if True, include `_system` pointing at the underlying nn.Module
                      (default: False)
    """

    model: dict[str, Any]
    input_key: str = "inputs"
    output_key: str = "logits"
    return_features: bool = False
    features_key: str = "features"
    include_system: bool = False

    def __post_init__(self) -> None:
        cfg = ModelConfig.model_validate(self.model)
        cfg = Lowerer().lower_model(cfg)
        Validator().validate_model_config(cfg)
        self.config = cfg
        self.module: nn.Module = Model(cfg)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "GenericSystem":
        self.module = self.module.to(device=device, dtype=dtype)
        return self

    def train(self, mode: bool = True) -> None:
        self.module.train(mode)

    def eval(self) -> None:
        self.module.eval()

    def forward(self, batch: dict[str, Any], *, ctx: object | None = None) -> dict[str, Any]:
        if self.input_key not in batch:
            raise KeyError(f"Missing batch key {self.input_key!r}")
        x = batch[self.input_key]
        if not isinstance(x, Tensor):
            raise TypeError(
                f"Expected batch[{self.input_key!r}] to be a Tensor, got {type(x).__name__}"
            )

        outputs: dict[str, Any] = {}
        if bool(self.return_features):
            result = self.module(x, ctx=ctx, return_features=True)  # type: ignore[call-arg]
            if not (isinstance(result, tuple) and len(result) == 2):
                raise TypeError("Expected Model(return_features=True) to return (features, out)")
            features, out = result
            outputs[str(self.features_key)] = features
            outputs[str(self.output_key)] = out
        else:
            out = self.module(x, ctx=ctx)  # type: ignore[call-arg]
            outputs[str(self.output_key)] = out

        if bool(self.include_system):
            outputs["_system"] = self.module
        return outputs

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)

