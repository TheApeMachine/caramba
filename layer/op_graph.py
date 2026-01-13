"""OpGraphLayer: execute a named-port operation graph as a single-stream layer.

This layer exists to make manifest-defined operation graphs usable inside the
standard Tensor->Tensor topology stack (Stacked/Residual/Nested/...).

Unlike `topology.GraphTopology`, this layer:
- Always accepts a single tensor stream (`x`) and returns a single tensor.
- Allows intermediate values to be non-tensors (e.g., cache objects or ints).
  This is important for making KV-cache logic composable in op graphs.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import TypeAdapter
from torch import Tensor, nn
from typing_extensions import override

from caramba.compiler.validate import Validator
from caramba.config.layer import OpGraphLayerConfig
from caramba.config.topology import GraphNodeConfig, GraphTopologyConfig, TopologyType
from caramba.topology.graph import _build_op, _call_op  # noqa: SLF001
from caramba.topology.utils import unwrap_output


def _as_list(x: str | list[str]) -> list[str]:
    return [x] if isinstance(x, str) else [str(v) for v in x]


class OpGraphLayer(nn.Module):
    """A Tensor->Tensor layer backed by a named-port operation graph."""

    def __init__(self, config: OpGraphLayerConfig) -> None:
        super().__init__()
        self.config = config

        payload: dict[str, Any] = dict(getattr(config, "graph", None) or {})
        # Support shorthand: if the payload omits the topology discriminator,
        # assume GraphTopology.
        payload.setdefault("type", TopologyType.GRAPH.value)

        topo = TypeAdapter(GraphTopologyConfig).validate_python(payload)

        # Expand simple per-node repeat (same behavior as GraphTopology).
        nodes: list[GraphNodeConfig] = []
        for n in topo.nodes:
            r = int(getattr(n, "repeat", 1) or 1)
            if r <= 1:
                nodes.append(n)
                continue
            in_keys = _as_list(n.in_keys)
            out_keys = _as_list(n.out_keys)
            if len(in_keys) != 1 or len(out_keys) != 1:
                raise ValueError(f"repeat only supported for single in/out keys (node {n.id!r})")
            src = in_keys[0]
            dst = out_keys[0]
            prev = src
            for i in range(r):
                cur_out = dst if i == (r - 1) else f"{dst}__{i}"
                nodes.append(
                    GraphNodeConfig(
                        id=f"{n.id}__{i}",
                        op=n.op,
                        **{"in": prev, "out": cur_out},
                        config=dict(n.config),
                        repeat=1,
                    )
                )
                prev = cur_out
        inputs = getattr(topo, "inputs", None)
        if inputs is not None:
            # OpGraphLayer provides these implicit inputs at runtime.
            s = {str(k) for k in list(inputs)}
            s.add(str(self.config.input_key))
            s.add("infer_ctx")
            inputs = sorted(s)
        topo = GraphTopologyConfig(type=topo.type, nodes=nodes, inputs=inputs)

        v = Validator()
        v.validate_graph_topology(topo, path="model.topology")
        self._order: list[str] = v.toposort_graph(topo)
        self._node_by_id: dict[str, GraphNodeConfig] = {n.id: n for n in topo.nodes}

        self.modules_by_id = nn.ModuleDict({n.id: _build_op(n.op, dict(n.config)) for n in topo.nodes})

    @override
    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        streams: dict[str, Any] = {str(self.config.input_key): x, "infer_ctx": ctx}

        for node_id in self._order:
            n = self._node_by_id.get(node_id)
            if n is None:
                raise KeyError(f"OpGraphLayer: missing node id {node_id!r} in topology")
            ins = _as_list(n.in_keys)
            outs = _as_list(n.out_keys)
            args: list[Any] = []
            for k in ins:
                if k not in streams:
                    raise KeyError(f"OpGraphLayer: missing input key {k!r} for node {n.id!r}")
                args.append(streams[k])

            mod = self.modules_by_id[n.id]
            out = _call_op(mod, ins, args, ctx=ctx)  # type: ignore[arg-type]

            if len(outs) == 1:
                if isinstance(out, tuple):
                    out = unwrap_output(out)
                elif isinstance(out, list) and out:
                    out = out[0]
                streams[outs[0]] = out
            else:
                if not isinstance(out, (tuple, list)) or len(out) != len(outs):
                    raise TypeError(f"Node {n.id!r} expected {len(outs)} outputs")
                for k, v in zip(outs, out, strict=True):
                    streams[k] = v

        y = streams.get(str(self.config.output_key), None)
        if not isinstance(y, Tensor):
            raise TypeError(
                f"OpGraphLayer expected Tensor output at key {self.config.output_key!r}, got {type(y).__name__}"
            )
        return y
