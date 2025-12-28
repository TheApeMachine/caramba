"""Transform configs for TensorDict-based data pipelines."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field


class RenameKeysTransformConfig(BaseModel):
    type: Literal["rename_keys"] = "rename_keys"
    mapping: dict[str, str]


class CastDtypeTransformConfig(BaseModel):
    type: Literal["cast_dtype"] = "cast_dtype"
    # key -> dtype string (torch dtype name, e.g. float32, float16, int64)
    dtypes: dict[str, str]


class AddMaskTransformConfig(BaseModel):
    type: Literal["add_mask"] = "add_mask"
    src_key: str = "targets"
    mask_key: str = "mask"
    # mask = (src != ignore_index)
    ignore_index: int = -100


class TokenShiftTransformConfig(BaseModel):
    type: Literal["token_shift"] = "token_shift"
    src_key: str = "tokens"
    input_key: str = "input_ids"
    target_key: str = "target_ids"


class GaussianNoiseTransformConfig(BaseModel):
    type: Literal["gaussian_noise"] = "gaussian_noise"
    src_key: str = "inputs"
    out_key: str = "noisy"
    sigma: float = 1.0


class GraphBatchTransformConfig(BaseModel):
    """Placeholder for future graph batching (no-op today)."""

    type: Literal["graph_batch"] = "graph_batch"


TransformConfig: TypeAlias = Annotated[
    RenameKeysTransformConfig
    | CastDtypeTransformConfig
    | AddMaskTransformConfig
    | TokenShiftTransformConfig
    | GaussianNoiseTransformConfig
    | GraphBatchTransformConfig,
    Field(discriminator="type"),
]


class TransformPipelineConfig(BaseModel):
    transforms: list[TransformConfig] = Field(default_factory=list)

