"""Lowering pass: expand user shortcuts to canonical configs.

Configs often use shortcuts for brevity (e.g., `repeat: 16` instead of
listing 16 layers). The lowerer expands these so downstream code sees
fully explicit, uniform structures.
"""
from __future__ import annotations

from config.manifest import Manifest
from config.model import ModelConfig
from config.target import ExperimentTargetConfig
from config.run import Run
from config.topology import NodeConfig, TopologyConfig
from config.topology import GraphNodeConfig, GraphTopologyConfig


class Lowerer:
    """Expands repeat declarations at compile-time.

    Transforms `repeat: N` into N explicit copies of the repeated structure,
    making the config canonical for model construction.
    """

    def lower_manifest(self, manifest: Manifest) -> Manifest:
        """Lower a manifest into canonical form.

        In manifest v2 we lower component configs where we recognize canonical
        structures. Currently:
        - `system.language_model` with `config.model` (a ModelConfig payload)
        """
        lowered_targets = []
        for t in list(getattr(manifest, "targets", [])):
            # Only experiment targets have runs to lower.
            if isinstance(t, ExperimentTargetConfig):
                t2 = t.model_copy(deep=True)

                # Lower multi-seed runs to concrete runs.
                t2.runs = self.lower_runs(list(t2.runs))

                # Lower model topology/geometry where applicable.
                if t2.system.ref in ("system.language_model", "system.generic"):
                    model_payload = t2.system.config.get("model", None)
                    if isinstance(model_payload, dict):
                        cfg = ModelConfig.model_validate(model_payload)
                        cfg = cfg.resolve_geometry()
                        lowered = self.lower_model(cfg)
                        t2.system.config["model"] = lowered.model_dump(by_alias=True)

                lowered_targets.append(t2)
                continue

            lowered_targets.append(t)

        return manifest.model_copy(update={"targets": lowered_targets})

    def lower_runs(self, runs: list[Run]) -> list[Run]:
        """Expand multi-seed runs into concrete runs.

        Allows YAML like:
          seed: [1, 2, 3]

        to expand into three runs with stable ids and experiment names:
          id: <id>__s1, exp: <exp>__s1, seed: 1
          id: <id>__s2, exp: <exp>__s2, seed: 2
          id: <id>__s3, exp: <exp>__s3, seed: 3
        """
        out: list[Run] = []
        for r in runs:
            seed = getattr(r, "seed", 0)
            if isinstance(seed, list):
                if not seed:
                    raise ValueError(f"run {r.id!r} provided empty seed list")
                for s in seed:
                    si = int(s)
                    rr = r.model_copy(deep=True)
                    rr.seed = si
                    rr.id = f"{r.id}__s{si}"
                    rr.exp = f"{r.exp}__s{si}"
                    out.append(rr)
            else:
                rr = r.model_copy(deep=True)
                rr.seed = int(seed)
                out.append(rr)
        return out

    def lower_model(self, model: ModelConfig) -> ModelConfig:
        """Lower a model config into canonical form."""
        lowered_topology = self.lower_topology(model.topology)
        return model.model_copy(update={"topology": lowered_topology})

    def lower_topology(self, config: TopologyConfig) -> TopologyConfig:
        """Expand topology-level repeat.

        Recursively lowers child nodes, then replicates the result
        according to the repeat count.
        """
        if isinstance(config, GraphTopologyConfig):
            return self.lower_graph_topology(config)
        lowered = self.lower_nodes(list(config.layers))
        layers = self.repeat_nodes(lowered, repeat=int(config.repeat))
        return config.model_copy(update={"layers": layers, "repeat": 1})

    def lower_graph_topology(self, topo: GraphTopologyConfig) -> GraphTopologyConfig:
        """Expand simple per-node repeat for graph topologies."""
        out: list[GraphNodeConfig] = []
        for n in topo.nodes:
            r = int(getattr(n, "repeat", 1) or 1)
            if r <= 1:
                out.append(n)
                continue
            ins = [str(n.in_keys)] if isinstance(n.in_keys, str) else [str(x) for x in n.in_keys]
            outs = [str(n.out_keys)] if isinstance(n.out_keys, str) else [str(x) for x in n.out_keys]
            if len(ins) != 1 or len(outs) != 1:
                raise ValueError(f"graph node repeat requires single in/out keys (node={n.id})")
            src = ins[0]
            dst = outs[0]
            prev = src
            for i in range(r):
                cur_out = dst if i == (r - 1) else f"{dst}__{i}"
                out.append(
                    GraphNodeConfig(
                        id=f"{n.id}__{i}",
                        op=n.op,
                        **{"in": prev, "out": cur_out},
                        config=dict(n.config),
                        repeat=1,
                    )
                )
                prev = cur_out
        return GraphTopologyConfig(type=topo.type, nodes=out, inputs=getattr(topo, "inputs", None))

    def lower_nodes(self, nodes: list[NodeConfig]) -> list[NodeConfig]:
        """Lower nested topology nodes recursively."""
        return [
            self.lower_topology(node) if self.is_topology(node) else node  # type: ignore[arg-type]
            for node in nodes
        ]

    def repeat_nodes(self, nodes: list[NodeConfig], *, repeat: int) -> list[NodeConfig]:
        """Repeat nodes by deep-copying each element.

        Deep copy ensures each layer has independent parameters.
        """
        return [node.model_copy(deep=True) for _ in range(repeat) for node in nodes]

    def is_topology(self, node: NodeConfig) -> bool:
        """Check if node is a topology (has layers list attribute)."""
        if isinstance(node, GraphTopologyConfig):
            return True
        if not hasattr(node, "layers"):
            return False
        layers = getattr(node, "layers")
        return isinstance(layers, list)
