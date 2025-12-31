from __future__ import annotations

import torch
from tensordict import TensorDict

from caramba.runtime.tensordict_utils import TensorDictBase, to_device


def test_to_device_preserves_tensordict_type() -> None:
    td = TensorDict({"x": torch.zeros(2, 3)}, batch_size=[2])
    out = to_device(td, device=torch.device("cpu"))
    assert isinstance(out, TensorDictBase)


def test_to_device_preserves_tensordict_inside_mapping() -> None:
    td = TensorDict({"x": torch.zeros(2, 3)}, batch_size=[2])
    out = to_device({"a": td}, device=torch.device("cpu"))
    assert isinstance(out, dict)
    assert isinstance(out["a"], TensorDictBase)

