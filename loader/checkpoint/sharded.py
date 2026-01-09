"""Sharded checkpoint loader.

Large checkpoints are often split across multiple files with an `.index.json`
file describing which tensor lives in which shard. This loader reads the index
and merges all shards into a single in-memory state_dict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from torch import Tensor

from caramba.loader.checkpoint.base import Checkpoint, StateDict
from caramba.loader.checkpoint.error import CheckpointError, CheckpointErrorType

if TYPE_CHECKING:
    from caramba.loader.checkpoint.builder import CheckpointBuilder


class CheckpointSharded(Checkpoint):
    """Sharded checkpoint loader.

    Consumes an `.index.json` file and loads all referenced shards.
    """
    def __init__(self, builder: CheckpointBuilder | None = None) -> None:
        if builder is None:
            from caramba.loader.checkpoint.builder import CheckpointBuilder as CB
            self.builder = CB()
        else:
            self.builder = builder

    def load(self, path: Path) -> StateDict:
        index_path = Path(path)
        if not index_path.exists():
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_NOT_FOUND)

        if index_path.is_dir():
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_IS_DIRECTORY)

        if not index_path.name.endswith(".index.json"):
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_INVALID_FORMAT)

        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")

        if not isinstance(weight_map, dict):
            raise CheckpointError(CheckpointErrorType.CHECKPOINT_INVALID_FORMAT)

        out: dict[str, Tensor] = {}

        for shard_name in sorted(set(weight_map.values())):
            shard_path = index_path.parent / str(shard_name)
            shard = self.builder.build(shard_path).load(shard_path)

            for key, value in shard.items():
                if key in out:
                    raise CheckpointError(CheckpointErrorType.CHECKPOINT_DUPLICATE_KEY)

                out[str(key)] = value

        return out

