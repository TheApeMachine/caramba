from __future__ import annotations

import pytest

from compiler.lower import Lowerer
from config.topology_graph import GraphNodeConfig, GraphTopologyConfig


def test_lower_graph_topology_expands_repeat_with_chained_keys() -> None:
    topo = GraphTopologyConfig(
        type="GraphTopology",
        layers=[
            GraphNodeConfig(
                id="n",
                op="mlp",
                **{"in": "x", "out": "y"},
                repeat=3,
                config={"d_in": 4, "d_out": 4},
            )
        ],
    )
    lowered = Lowerer().lower_graph_topology(topo)
    assert len(lowered.layers) == 3
    assert lowered.layers[0].id == "n__0"
    assert lowered.layers[0].in_keys == "x"
    assert lowered.layers[0].out_keys == "y__0"
    assert lowered.layers[1].in_keys == "y__0"
    assert lowered.layers[1].out_keys == "y__1"
    assert lowered.layers[2].in_keys == "y__1"
    assert lowered.layers[2].out_keys == "y"


def test_lower_graph_topology_repeat_requires_single_in_out_keys() -> None:
    topo = GraphTopologyConfig(
        type="GraphTopology",
        layers=[
            GraphNodeConfig(
                id="n",
                op="mlp",
                **{"in": ["a", "b"], "out": "y"},
                repeat=2,
                config={},
            )
        ],
    )
    with pytest.raises(ValueError, match="single in/out keys"):
        Lowerer().lower_graph_topology(topo)
