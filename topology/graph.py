"""Graph topology: named-port DAG executed over a TensorDict/dict.

This is the unified topology-level implementation for graph models. It executes
`GraphTopologyConfig` nodes in topological order, reading tensors from named
keys and writing tensors back to named keys.
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from typing import Any

from pydantic import TypeAdapter
import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.compiler.validate import Validator
from caramba.config.layer import LayerConfig, LayerType
from caramba.config.topology import GraphNodeConfig, GraphTopologyConfig
from caramba.topology.utils import unwrap_output
from caramba.runtime.tensordict_utils import TensorDictBase, as_tensordict


def _as_list(x: str | list[str]) -> list[str]:
    return [x] if isinstance(x, str) else [str(v) for v in x]


def _import_symbol(path: str) -> object:
    """Import a symbol from a `module:Symbol` string."""
    if ":" not in path:
        raise ValueError(f"Invalid python symbol path {path!r}. Expected 'module:Symbol'.")
    mod_name, sym = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, sym)
    except AttributeError as e:
        raise ImportError(f"Symbol {sym!r} not found in module {mod_name!r}") from e


def _build_op(op: str, cfg: dict[str, Any]) -> nn.Module:
    """Build a node op from a string identifier.

    Supported:
    - `python:module:Symbol` where Symbol constructs an nn.Module
    - `torch.nn.<OpName>` shorthand: just pass `<OpName>` (e.g. Linear, Conv2d)
    - Caramba layer config types: pass the LayerType value (e.g. LinearLayer, Conv2dLayer)
    - Caramba operation types: pass an Operation class name exported by `caramba.operation`
    """
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

    # Caramba layers (config-driven). This makes graph nodes feel like the rest
    # of the platform: you specify a stable `type` and a config payload.
    try:
        _ = LayerType.from_str(s)
    except ValueError:
        # Not a Caramba layer type — fall through to torch.nn.
        pass
    else:
        layer_cfg = TypeAdapter(LayerConfig).validate_python({"type": s, **dict(cfg)})
        return layer_cfg.build()  # type: ignore[attr-defined]

    # Caramba operations (manifest-friendly building blocks).
    try:
        import caramba.operation as _ops

        sym = getattr(_ops, s, None)
        if isinstance(sym, type) and issubclass(sym, nn.Module):
            return sym(**cfg)  # type: ignore[call-arg]
    except Exception:
        pass

    # Torch built-ins (nn.Linear, nn.Conv2d, etc.)
    if hasattr(nn, s):
        cls = getattr(nn, s)
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            return cls(**cfg)  # type: ignore[call-arg]

    raise KeyError(
        f"Unknown op {op!r}. Use a Caramba LayerType (e.g. LinearLayer), torch.nn.<OpName>, "
        "or python:module:Symbol"
    )


@lru_cache(maxsize=512)
def _forward_param_names(cls: type[nn.Module]) -> tuple[str, ...] | None:
    """Best-effort extraction of forward() param names for signature-based calls."""
    try:
        sig = inspect.signature(cls.forward)
    except (TypeError, ValueError):
        return None

    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]

    names: list[str] = []
    for p in params:
        # ctx is supplied out-of-band by GraphTopology/OpGraphLayer.
        if p.name == "ctx":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return None
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            return None
        names.append(str(p.name))

    return tuple(names)


def _call_op_signature_fallback(mod: nn.Module, args: list[Any], *, ctx: object | None) -> object | None:
    """Attempt calling an op by mapping args to forward() param names."""
    names = _forward_param_names(type(mod))
    if not names:
        return None
    if len(args) > len(names):
        return None
    kwargs = {k: v for k, v in zip(list(names)[: len(args)], args, strict=True)}
    if ctx is not None:
        try:
            return mod(**kwargs, ctx=ctx)
        except TypeError:
            pass
    try:
        return mod(**kwargs)
    except TypeError:
        return None


def _call_op(mod: nn.Module, arg_names: list[str], args: list[Any], *, ctx: object | None) -> object:
    """Call a node op with argument adapters.

    Adapters we apply (in order):
    - Prefer keyword calls using graph port names (`in:` keys) so operation-style
      modules can use keyword-only signatures (e.g. `forward(*, x=...)`).
    - Pass `ctx=...` if the op accepts it.
    - Fall back to positional calls.
    - If the op doesn't accept multiple positional inputs, fall back to passing
      a single tuple (useful for layers like GraphConv that expect (x, adj)).
    """
    if arg_names and len(arg_names) == len(args):
        kwargs = {k: v for k, v in zip(arg_names, args, strict=True)}
        if ctx is not None:
            try:
                return mod(**kwargs, ctx=ctx)
            except TypeError:
                pass
        try:
            return mod(**kwargs)
        except TypeError:
            pass

    if ctx is not None:
        try:
            return mod(*args, ctx=ctx) if len(args) != 1 else mod(args[0], ctx=ctx)
        except TypeError:
            pass

    try:
        return mod(*args) if len(args) != 1 else mod(args[0])
    except TypeError:
        out = _call_op_signature_fallback(mod, args, ctx=ctx)
        if out is not None:
            return out
        if len(args) <= 1:
            raise
        # Fallback: pack inputs into a single tuple.
        if ctx is not None:
            try:
                return mod(tuple(args), ctx=ctx)
            except TypeError:
                pass
        return mod(tuple(args))


class GraphTopology(nn.Module):
    """Execute a named-port DAG over a TensorDict/dict."""

    def __init__(self, config: GraphTopologyConfig) -> None:
        super().__init__()
        self.config = config

        # Expand simple repeats (compiler lowering can also do this).
        nodes: list[GraphNodeConfig] = []
        for n in config.nodes:
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
        topo = GraphTopologyConfig(type=config.type, nodes=nodes, inputs=getattr(config, "inputs", None))

        v = Validator()
        v.validate_graph_topology(topo, path="model.topology")
        self._order: list[str] = v.toposort_graph(topo)
        self._node_by_id: dict[str, GraphNodeConfig] = {n.id: n for n in topo.nodes}

        self.nodes = topo.nodes
        self.modules_by_id = nn.ModuleDict(
            {n.id: _build_op(n.op, dict(n.config)) for n in topo.nodes}
        )

    def to(self, *, device: torch.device, dtype: torch.dtype) -> GraphTopology:  # type: ignore[override]
        self.modules_by_id = self.modules_by_id.to(device=device, dtype=dtype)
        return self

    @override
    def forward(self, batch: TensorDictBase | dict[str, Any], *, ctx: object | None = None) -> TensorDictBase:
        streams: dict[str, Any] = dict(batch)

        for node_id in self._order:
            n = self._node_by_id.get(node_id)
            if n is None:
                raise KeyError(f"GraphTopology: missing node id {node_id!r} in topology")
            ins = _as_list(n.in_keys)
            outs = _as_list(n.out_keys)
            args: list[Tensor] = []
            for k in ins:
                v = streams.get(k, None)
                if not isinstance(v, Tensor):
                    raise KeyError(
                        f"GraphTopology: missing tensor input key {k!r} for node {n.id!r}"
                    )
                args.append(v)
            mod = self.modules_by_id[n.id]
            out = _call_op(mod, ins, args, ctx=ctx)
            if len(outs) == 1:
                if isinstance(out, tuple):
                    out = unwrap_output(out)
                elif isinstance(out, list) and out:
                    out = out[0]
                if not isinstance(out, Tensor):
                    raise TypeError(
                        f"Node {n.id!r} expected Tensor output, got {type(out).__name__}"
                    )
                streams[outs[0]] = out
            else:
                if not isinstance(out, (tuple, list)) or len(out) != len(outs):
                    raise TypeError(f"Node {n.id!r} expected {len(outs)} outputs")
                for k, v in zip(outs, out, strict=True):
                    if not isinstance(v, Tensor):
                        raise TypeError(f"Node {n.id!r} output {k!r} is not a Tensor")
                    streams[k] = v

        return as_tensordict(streams)
