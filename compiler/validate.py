from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeGuard

from config.model import ModelConfig
from config.topology_graph import GraphTopologyConfig
from config.layer import (
    AttentionLayerConfig,
    AttentionMode,
    DropoutLayerConfig,
    LayerConfig,
    LinearLayerConfig,
)
from config.topology import NodeConfig, TopologyConfig
from console import logger

from compiler.plan import Planner


@dataclass(frozen=True, slots=True)
class IO:
    d_in: int | None = None
    d_out: int | None = None


class Validator:
    def validate_model_config(self, model: ModelConfig, *, print_plan: bool = False) -> ModelConfig:
        """Validate a model config's topology/layer invariants."""
        topo = model.topology
        self.infer_topology_io(topo, path="model.topology")
        for layer, path in self.iter_layers(topo, path="model.topology"):
            if isinstance(layer, AttentionLayerConfig):
                self.validate_attention(layer, path=path)

        # Planner currently only formats legacy manifests; keep flag for parity.
        _ = print_plan
        return model

    def validate_manifest(self, manifest: object, *, print_plan: bool = False) -> object:
        """Legacy shim: validators are now component-specific.

        Manifest v2 is validated by the target runner/engine based on referenced components.
        """
        _ = print_plan
        return manifest

    validate = validate_manifest

    # -----------------------------
    # Graph topology validation (TensorDict DAGs)
    # -----------------------------

    def validate_graph_topology(self, topo: GraphTopologyConfig, *, path: str = "system.topology") -> None:
        ids: set[str] = set()
        produced: dict[str, str] = {}
        for i, n in enumerate(list(topo.nodes)):
            p = f"{path}.nodes[{i}]"
            nid = str(n.id)
            if not nid:
                raise ValueError(f"{p}.id: must be non-empty")
            if nid in ids:
                raise ValueError(f"{p}.id: duplicate node id {nid!r}")
            ids.add(nid)

            outs = [str(n.out_keys)] if isinstance(n.out_keys, str) else [str(x) for x in n.out_keys]
            if not outs:
                raise ValueError(f"{p}.out: must be non-empty")
            for k in outs:
                if k in produced:
                    raise ValueError(f"{p}.out: key {k!r} already produced by node {produced[k]!r}")
                produced[k] = nid

        # Acyclicity and ordering are validated by toposort_graph().
        _ = self.toposort_graph(topo)

    def toposort_graph(self, topo: GraphTopologyConfig) -> list[str]:
        nodes = list(topo.nodes)
        ids = [str(n.id) for n in nodes]
        produced: dict[str, str] = {}
        for n in nodes:
            outs = [str(n.out_keys)] if isinstance(n.out_keys, str) else [str(x) for x in n.out_keys]
            for k in outs:
                produced[k] = str(n.id)

        # Build adjacency using key dependencies.
        edges: dict[str, set[str]] = {nid: set() for nid in ids}
        indeg: dict[str, int] = {nid: 0 for nid in ids}
        for n in nodes:
            nid = str(n.id)
            ins = [str(n.in_keys)] if isinstance(n.in_keys, str) else [str(x) for x in n.in_keys]
            for k in ins:
                src = produced.get(k)
                if src is None or src == nid:
                    continue
                if nid not in edges[src]:
                    edges[src].add(nid)
                    indeg[nid] += 1

        # Kahn's algorithm.
        q = [nid for nid in ids if indeg[nid] == 0]
        out: list[str] = []
        while q:
            nid = q.pop()
            out.append(nid)
            for dst in edges.get(nid, ()):
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    q.append(dst)

        if len(out) != len(ids):
            # Find a small witness set (nodes still with indeg>0).
            stuck = [nid for nid in ids if indeg[nid] > 0][:8]
            raise ValueError(f"Graph topology contains a cycle or unresolved deps (stuck={stuck})")
        return out

    def validate_topology(self, config: TopologyConfig, *, path: str = "model.topology") -> None:
        self.infer_topology_io(config, path=path)

    def is_topology(self, node: NodeConfig) -> TypeGuard[TopologyConfig]:
        return hasattr(node, "layers")

    def infer_node_io(self, node: NodeConfig, *, path: str) -> IO:
        return self.infer_topology_io(
            node, path=path
        ) if self.is_topology(node) else self.infer_layer_io(node)  # type: ignore[arg-type]

    def infer_layer_io(self, layer: LayerConfig) -> IO:
        from config.layer import LinearLayerConfig, LoRALinearLayerConfig, DropoutLayerConfig

        if isinstance(layer, (LinearLayerConfig, LoRALinearLayerConfig)):
            return IO(int(layer.d_in), int(layer.d_out))
        if isinstance(layer, DropoutLayerConfig):
            return IO()
        d_model = getattr(layer, "d_model", None)
        if d_model is not None:
            d = int(d_model)
            return IO(d, d)
        raise ValueError(f"Unsupported layer config: {type(layer)!r}")

    def infer_topology_io(self, topo: TopologyConfig, *, path: str) -> IO:
        t = topo.type.value
        nodes = list(topo.layers)

        if t in ("ParallelTopology", "BranchingTopology"):
            outs: set[int] = set()
            for i, n in enumerate(nodes):
                d = self.infer_node_io(n, path=f"{path}.layers[{i}]").d_out
                if d is not None:
                    outs.add(d)
            return IO(d_out=self.single_dim(outs, path=path, kind=t))

        io = self.infer_seq_io(nodes, path=path)

        if t == "ResidualTopology" and io.d_out is not None:
            self.require_shape_preserving(nodes, path=path)

        return io

    def infer_seq_io(self, nodes: list[NodeConfig], *, path: str) -> IO:
        cur: int | None = None
        for i, node in enumerate(nodes):
            p = f"{path}.layers[{i}]"
            io = self.infer_node_io(node, path=p)

            if io.d_in is not None:
                if cur is not None and cur != io.d_in:
                    raise ValueError(
                        f"{p}: expected d_in={cur}, got d_in={io.d_in}. "
                        "Fix: make this node's input dim match the previous node's output dim."
                    )
                cur = io.d_in

            if io.d_out is not None:
                cur = io.d_out

        return IO(d_out=cur)

    def single_dim(self, dims: set[int], *, path: str, kind: str) -> int | None:
        if len(dims) > 1:
            raise ValueError(
                f"{path}: {kind} requires consistent d_out, got {sorted(dims)}. "
                f"Fix: ensure all {kind} branches produce the same d_out."
            )
        return next(iter(dims), None)

    def require_shape_preserving(self, nodes: list[NodeConfig], *, path: str) -> None:
        for i, node in enumerate(nodes):
            p = f"{path}.layers[{i}]"
            io = self.infer_node_io(node, path=p)
            if io.d_in is not None and io.d_out is not None and io.d_in != io.d_out:
                raise ValueError(
                    f"{p}: residual requires shape-preserving nodes, got d_in={io.d_in}, d_out={io.d_out}. "
                    "Fix: ensure all nodes inside residual preserve d_model."
                )

    def iter_layers(self, node: NodeConfig, *, path: str) -> Iterable[tuple[LayerConfig, str]]:
        if self.is_topology(node):
            for i, child in enumerate(node.layers):
                yield from self.iter_layers(child, path=f"{path}.layers[{i}]")
        else:
            yield node, path  # type: ignore[misc]

    def validate_attention(self, cfg: AttentionLayerConfig, *, path: str) -> None:
        def fail(msg: str) -> None:
            raise ValueError(f"{path}: {msg}")

        nh, nk = cfg.n_heads, cfg.kv_heads
        if nh % nk:
            fail(f"n_heads ({nh}) must be divisible by n_kv_heads ({nk})")

        if cfg.mode == AttentionMode.DECOUPLED:
            sd, gd = cfg.sem_dim, cfg.geo_dim
            if sd is None or gd is None:
                fail("decoupled mode requires sem_dim and geo_dim")
                return

            if sd % nh:
                fail(f"sem_dim ({sd}) must be divisible by n_heads ({nh})")
            if gd % nh:
                fail(f"geo_dim ({gd}) must be divisible by n_heads ({nh})")
            if cfg.v_dim % nh:
                fail(f"v_dim ({cfg.v_dim}) must be divisible by n_heads ({nh})")

            if cfg.rope_enabled:
                geo_head_dim = gd // nh
                if geo_head_dim % 2:
                    fail(f"geo_head_dim ({geo_head_dim}) must be even for RoPE")

        else:
            attn_dim = cfg.attn_dim or cfg.d_model
            if attn_dim != nh * cfg.head_dim:
                fail(f"attn_dim ({attn_dim}) must equal n_heads*head_dim ({nh * cfg.head_dim})")
