from __future__ import annotations

import pytest
import torch

from caramba.config.layer import LayerType, OpGraphLayerConfig


def test_op_graph_layer_builds_and_runs_keyword_only_ops() -> None:
    cfg = OpGraphLayerConfig(
        type=LayerType.OP_GRAPH,
        d_in=4,
        d_out=4,
        input_key="x",
        output_key="y",
        graph={
            "nodes": [
                {
                    "id": "scale",
                    "op": "ScaleOperation",
                    "in": "x",
                    "out": "y",
                    "config": {"scale": 2.0},
                }
            ],
            "inputs": ["x"],
        },
    )
    layer = cfg.build()

    x = torch.randn(2, 3, 4)
    y = layer(x)
    assert torch.allclose(y, x * 2.0)


def test_op_graph_layer_supports_multi_output_nodes() -> None:
    cfg = OpGraphLayerConfig(
        type=LayerType.OP_GRAPH,
        d_in=4,
        d_out=2,
        input_key="x",
        output_key="y",
        graph={
            "nodes": [
                {
                    "id": "split",
                    "op": "SplitOperation",
                    "in": "x",
                    "out": ["a", "b"],
                    "config": {"split_size": 2, "dim": -1},
                },
                {
                    "id": "add",
                    "op": "AddOperation",
                    "in": ["a", "b"],
                    "out": "y",
                    "config": {},
                },
            ],
            "inputs": ["x"],
        },
    )
    layer = cfg.build()

    x = torch.randn(1, 1, 4)
    y = layer(x)
    assert y.shape == (1, 1, 2)
    # y == first_half + second_half
    assert torch.allclose(y, x[..., :2] + x[..., 2:])


def test_op_graph_layer_raises_on_missing_output_key() -> None:
    cfg = OpGraphLayerConfig(
        type=LayerType.OP_GRAPH,
        d_in=4,
        d_out=4,
        input_key="x",
        output_key="y",
        graph={"nodes": [], "inputs": ["x"]},
    )
    layer = cfg.build()
    with pytest.raises(TypeError, match="expected Tensor output"):
        _ = layer(torch.randn(1, 1, 4))

