"""TensorDict graph system (DAG executor with named ports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from compiler.validate import Validator
from config.topology_graph import GraphNodeConfig, GraphTopologyConfig
from runtime.tensordict_utils import TensorDictBase, as_tensordict


def _as_list(x: str | list[str]) -> list[str]:
    return [x] if isinstance(x, str) else [str(v) for v in x]


def _import_symbol(path: str) -> object:
    if ":" not in path:
        raise ValueError(f"Invalid python symbol path {path!r}. Expected 'module:Symbol'.")
    mod_name, sym = path.split(":", 1)
    import importlib

    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, sym)
    except AttributeError as e:
        raise ImportError(f"Symbol {sym!r} not found in module {mod_name!r}") from e


def _build_op(op: str, cfg: dict[str, Any]) -> nn.Module:
    s = str(op)
    if s.startswith("python:"):
        factory = _import_symbol(s.removeprefix("python:"))
        if isinstance(factory, type) and issubclass(factory, nn.Module):
            return factory(**cfg)  # type: ignore[call-arg]
        if callable(factory):
            out = factory(**cfg)  # type: ignore[misc]
            if isinstance(out, nn.Module):
                return out
            raise TypeError(f"{s} did not construct an nn.Module")
        raise TypeError(f"{s} is not callable")

    # Torch built-ins (nn.Linear, nn.Conv2d, etc.)
    if hasattr(nn, s):
        cls = getattr(nn, s)
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            return cls(**cfg)  # type: ignore[call-arg]

    raise KeyError(f"Unknown op {op!r}. Use torch.nn.<OpName> or python:module:Symbol")


def _build_node(node: GraphNodeConfig) -> nn.Module:
    if node.layer is not None:
        return node.layer.build()
    if node.op is None:
        raise ValueError(f"Graph node {node.id!r} did not specify 'layer' or 'op'")
    return _build_op(node.op, dict(node.config))


@dataclass
class GraphSystem:
    """Execute a named-port DAG over a TensorDict.

    Config:
    - topology: GraphTopologyConfig payload (required)
    - output_keys: optional list of keys to return (default: all keys)
    """

    topology: dict[str, Any]
    output_keys: list[str] | None = None

    _nodes: list[GraphNodeConfig] = field(init=False, repr=False)
    _order: list[str] = field(init=False, repr=False)
    _node_by_id: dict[str, GraphNodeConfig] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        topo = GraphTopologyConfig.model_validate(self.topology)
        # Expand simple repeats (compiler lowering can also do this).
        nodes: list[GraphNodeConfig] = []
        for n in topo.layers:
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
                layer_cfg = n.layer.model_copy(deep=True) if n.layer is not None else None
                nodes.append(
                    GraphNodeConfig(
                        id=f"{n.id}__{i}",
                        in_keys=prev,
                        out_keys=cur_out,
                        layer=layer_cfg,
                        op=n.op,
                        config=dict(n.config),
                        repeat=1,
                    )
                )
                prev = cur_out
        topo = GraphTopologyConfig(type=topo.type, layers=nodes)

        Validator().validate_graph_topology(topo, path="system.topology")

        self._nodes = topo.layers
        self._order = Validator().toposort_graph(topo)
        self._node_by_id = {n.id: n for n in self._nodes}
        self.modules = nn.ModuleDict({n.id: _build_node(n) for n in self._nodes})

    def to(self, *, device: torch.device, dtype: torch.dtype) -> GraphSystem:
        self.modules = self.modules.to(device=device, dtype=dtype)
        return self

    def forward(self, batch: TensorDictBase, *, ctx: object | None = None) -> TensorDictBase:
        _ = ctx
        # Work in a plain dict for simplicity; return TensorDict at the end.
        streams: dict[str, Any] = dict(batch)

        for node_id in self._order:
            n = self._node_by_id.get(node_id)
            if n is None:
                raise KeyError(f"GraphSystem: missing node id {node_id!r} in topology")
            ins = _as_list(n.in_keys)
            outs = _as_list(n.out_keys)
            args: list[Tensor] = []
            for k in ins:
                v = streams.get(k, None)
                if not isinstance(v, Tensor):
                    raise KeyError(f"GraphSystem: missing tensor input key {k!r} for node {n.id!r}")
                args.append(v)
            mod = self.modules[n.id]
            out = mod(*args)
            if len(outs) == 1:
                if not isinstance(out, Tensor):
                    raise TypeError(f"Node {n.id!r} expected Tensor output, got {type(out).__name__}")
                streams[outs[0]] = out
            else:
                if not isinstance(out, (tuple, list)) or len(out) != len(outs):
                    raise TypeError(f"Node {n.id!r} expected {len(outs)} outputs")
                for k, v in zip(outs, out, strict=True):
                    if not isinstance(v, Tensor):
                        raise TypeError(f"Node {n.id!r} output {k!r} is not a Tensor")
                    streams[k] = v

        if self.output_keys is None:
            return as_tensordict(streams)
        return as_tensordict({k: streams[k] for k in self.output_keys if k in streams})

    def parameters(self):
        return self.modules.parameters()

    def state_dict(self):
        return self.modules.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.modules.load_state_dict(state)
