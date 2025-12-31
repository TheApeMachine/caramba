from __future__ import annotations

import pytest
import torch

from caramba.runtime.tensordict_utils import as_tensordict
from caramba.trainer.objectives import (
    KeyedCrossEntropyObjective,
    KeyedMSEObjective,
    NextTokenCrossEntropyObjective,
    _require_tensor,
)


def test_require_tensor_validates_presence_and_type() -> None:
    with pytest.raises(KeyError):
        _require_tensor({}, "x", where="outputs")
    with pytest.raises(TypeError):
        _require_tensor({"x": 1}, "x", where="batch")

    t = torch.randn(2, 3)
    assert _require_tensor({"x": t}, "x", where="outputs") is t


def test_keyed_mse_objective_loss_and_metrics() -> None:
    obj = KeyedMSEObjective(pred_key="p", target_key="t")
    outputs = as_tensordict({"p": torch.tensor([[1.0], [2.0]])})
    batch = as_tensordict({"t": torch.tensor([[0.0], [4.0]])})
    loss = obj.loss(outputs=outputs, batch=batch)
    assert float(loss) == pytest.approx(((1 - 0) ** 2 + (2 - 4) ** 2) / 2.0)
    m = obj.metrics(outputs=outputs, batch=batch, loss=loss)
    assert "mse" in m


def test_keyed_cross_entropy_objective_handles_shapes_and_ignore_index() -> None:
    obj = KeyedCrossEntropyObjective(logits_key="logits", labels_key="y", ignore_index=-100)
    logits = torch.zeros(2, 3)  # uniform => loss = log(C)
    labels = torch.tensor([0, -100])
    outputs = as_tensordict({"logits": logits})
    batch = as_tensordict({"y": labels})
    loss = obj.loss(outputs=outputs, batch=batch)
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(3.0))), rel=1e-6)
    m = obj.metrics(outputs=outputs, batch=batch, loss=loss)
    assert "ce_loss" in m


def test_next_token_ce_objective_aliases_target_key() -> None:
    obj = NextTokenCrossEntropyObjective(logits_key="l", target_key="targets")
    assert obj.labels_key == "targets"
    assert obj.target_key == "targets"

    logits = torch.zeros(2, 3)  # uniform => loss = log(C)
    targets = torch.tensor([0, 2])
    outputs = as_tensordict({"l": logits})
    batch = as_tensordict({"targets": targets})
    loss = obj.loss(outputs=outputs, batch=batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(3.0))), rel=1e-6)

