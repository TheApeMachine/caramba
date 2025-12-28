from __future__ import annotations

import torch

from model.graph_system import GraphSystem
from runtime.tensordict_utils import as_tensordict


def test_graph_system_executes_named_port_dag() -> None:
    sys = GraphSystem(
        topology={
            "type": "GraphTopology",
            "nodes": [
                {"id": "proj", "op": "Linear", "in": "inputs", "out": "h", "config": {"in_features": 4, "out_features": 8}},
                {"id": "act", "op": "ReLU", "in": "h", "out": "h2", "config": {}},
                {"id": "head", "op": "Linear", "in": "h2", "out": "logits", "config": {"in_features": 8, "out_features": 3}},
            ],
        },
        output_keys=["logits"],
    )

    batch = as_tensordict({"inputs": torch.randn(2, 4)})
    out = sys.forward(batch)
    assert "logits" in out
    assert tuple(out["logits"].shape) == (2, 3)


def test_graph_system_repeat_expands_single_stream() -> None:
    sys = GraphSystem(
        topology={
            "type": "GraphTopology",
            "nodes": [
                {"id": "proj", "op": "Linear", "in": "inputs", "out": "h", "repeat": 3, "config": {"in_features": 4, "out_features": 4}},
            ],
        },
        output_keys=["h"],
    )
    batch = as_tensordict({"inputs": torch.randn(5, 4)})
    out = sys.forward(batch)
    assert tuple(out["h"].shape) == (5, 4)

